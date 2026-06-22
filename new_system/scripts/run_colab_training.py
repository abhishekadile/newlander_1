#!/usr/bin/env python3
"""
run_colab_training.py
=====================
Phase 4: Orchestrate YOLO26 training on a Google Colab T4 GPU via the
official Colab CLI (https://github.com/googlecolab/google-colab-cli).

This script is written in Python so it runs on Windows, macOS, and Linux
without requiring a bash shell. The colab CLI must be installed and accessible:
    pip install google-colab-cli  (or)  uv tool install google-colab-cli

Usage:
    cd new_system/
    python scripts/run_colab_training.py [--session mycolony] [--no-drive]

What this script does:
  1. colab new --gpu T4          -- provision GPU session
  2. colab status                -- verify T4 was assigned (fail if CPU)
  3. colab drivemount            -- mount Google Drive for persistence
  4. Upload processed data + configs to Drive or session
  5. colab install ultralytics   -- install deps on VM
  6. colab exec -f train script  -- run training (with high timeout)
  7. Periodic checkpoint download every ~30 min (background thread)
  8. colab download best.pt      -- final weights retrieval
  9. colab log -o session.ipynb  -- archive session notebook
 10. colab stop                  -- release VM (always, even on error)

IMPORTANT: Google OAuth login is required on first run. The script will
pause and prompt you to complete the browser authentication flow.
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent      # new_system/
REPO_ROOT = ROOT.parent       # d:/work/cell/

WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"
DATA_PROC = ROOT / "data" / "processed"
CONFIGS = ROOT / "configs"

# Remote paths on Colab VM
REMOTE_ROOT = "/content/colony_project"
REMOTE_DATA = f"{REMOTE_ROOT}/data/processed"
REMOTE_CONFIGS = f"{REMOTE_ROOT}/configs"
REMOTE_RUNS = f"{REMOTE_ROOT}/runs"
REMOTE_WEIGHTS = f"{REMOTE_ROOT}/weights"

# Checkpoint download interval (seconds)
CHECKPOINT_INTERVAL = 1800  # 30 min

# Training timeout — 150 epochs × ~2 min/epoch on T4 ≈ 5 hours; set 8h headroom
TRAINING_TIMEOUT = 8 * 3600  # 8 hours in seconds


def run(cmd: list, check: bool = True, capture: bool = False, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a colab CLI command and return the result."""
    full_cmd = cmd if isinstance(cmd, list) else cmd.split()
    print(f"\n  $ {' '.join(str(c) for c in full_cmd)}")
    try:
        result = subprocess.run(
            full_cmd,
            check=check,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
        if capture and result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: command failed (exit {e.returncode})", file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        raise
    except subprocess.TimeoutExpired:
        print(f"  ERROR: command timed out after {timeout}s", file=sys.stderr)
        raise


def colab(*args, check: bool = True, capture: bool = False, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a `colab` subcommand."""
    return run(["colab"] + list(str(a) for a in args), check=check, capture=capture, timeout=timeout)


def stop_session(session_name: str):
    """Best-effort session cleanup."""
    print("\n  [cleanup] Stopping Colab session …")
    try:
        colab("stop", "-s", session_name, check=False, timeout=30)
    except Exception as e:
        print(f"  WARNING: colab stop failed: {e}", file=sys.stderr)


def checkpoint_loop(session_name: str, remote_last: str, local_last: Path, stop_event: threading.Event):
    """Background thread: download last.pt every CHECKPOINT_INTERVAL seconds."""
    while not stop_event.wait(CHECKPOINT_INTERVAL):
        print(f"\n  [checkpoint] Downloading last.pt from remote …")
        try:
            local_last.parent.mkdir(parents=True, exist_ok=True)
            colab(
                "download", remote_last,
                "-s", session_name,
                check=False, timeout=120
            )
            # The colab download command saves to the local working directory
            # Move it to weights/ if needed
            downloaded = Path(Path(remote_last).name)
            if downloaded.exists():
                import shutil
                shutil.move(str(downloaded), str(local_last))
                print(f"  [checkpoint] Saved to {local_last}")
        except Exception as e:
            print(f"  [checkpoint] Download failed: {e}", file=sys.stderr)


def make_training_stub(args) -> Path:
    """
    Write a small stub script that sets up paths and runs train_colony_model.py
    on the Colab VM where the working directory is REMOTE_ROOT.
    """
    stub = f"""#!/usr/bin/env python3
import sys, subprocess, os
from pathlib import Path

# Set up remote paths
os.chdir("{REMOTE_ROOT}")
sys.path.insert(0, "{REMOTE_ROOT}/scripts")

# Check GPU
import subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True)
print("[GPU]", gpu.stdout.strip())

# Install deps
subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics", "-q"])

# Run training
train_script = "{REMOTE_ROOT}/scripts/train_colony_model.py"
train_args = [sys.executable, train_script,
              "--device", "0",
              "--epochs", "{args.epochs}",
              "--imgsz", "{args.imgsz}",
              "--batch", "{args.batch}",
              "--workers", "4"]
print("[TRAIN] Running:", " ".join(train_args))
result = subprocess.run(train_args)
sys.exit(result.returncode)
"""
    tf = Path(tempfile.gettempdir()) / "colony_train_stub.py"
    tf.write_text(stub, encoding="utf-8")
    return tf


def upload_dataset(session_name: str, use_drive: bool):
    """Upload processed data and configs to the Colab VM."""
    print("\n=== Uploading dataset to Colab VM ===")
    # Create remote directories via exec
    setup_code = f"""
import os, pathlib
for d in [
    "{REMOTE_DATA}/images/train",
    "{REMOTE_DATA}/images/val",
    "{REMOTE_DATA}/images/test",
    "{REMOTE_DATA}/labels/train",
    "{REMOTE_DATA}/labels/val",
    "{REMOTE_DATA}/labels/test",
    "{REMOTE_CONFIGS}",
    "{REMOTE_RUNS}",
    "{REMOTE_WEIGHTS}",
    "{REMOTE_ROOT}/scripts",
]:
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
print("Directories created.")
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(setup_code)
        setup_script = Path(tf.name)

    try:
        colab("exec", "-f", str(setup_script), "-s", session_name, timeout=60)
    finally:
        setup_script.unlink(missing_ok=True)

    # Upload configs
    colab("upload", str(CONFIGS / "data.yaml"),
          f"{REMOTE_CONFIGS}/data.yaml", "-s", session_name, timeout=120)

    # Upload training script
    colab("upload", str(SCRIPT_DIR / "train_colony_model.py"),
          f"{REMOTE_ROOT}/scripts/train_colony_model.py", "-s", session_name, timeout=120)

    # Upload images and labels (per-split)
    for split in ("train", "val", "test"):
        img_dir = DATA_PROC / "images" / split
        lbl_dir = DATA_PROC / "labels" / split

        if not img_dir.exists():
            print(f"  WARNING: {img_dir} does not exist, skipping {split} split.")
            continue

        img_files = list(img_dir.iterdir())
        print(f"  Uploading {len(img_files)} {split} images …")

        # For large datasets, provide guidance on Drive-based approach
        total_mb = sum(f.stat().st_size for f in img_files if f.is_file()) / (1024 * 1024)
        if total_mb > 2000 and not use_drive:
            print(f"  WARNING: {total_mb:.0f} MB of {split} images is large.")
            print("  Consider using --drive-path to upload to mounted Google Drive instead.")

        for f in img_files:
            if f.is_file():
                colab("upload", str(f),
                      f"{REMOTE_DATA}/images/{split}/{f.name}",
                      "-s", session_name, check=False, timeout=120)

        lbl_files = list(lbl_dir.iterdir()) if lbl_dir.exists() else []
        print(f"  Uploading {len(lbl_files)} {split} labels …")
        for f in lbl_files:
            if f.is_file():
                colab("upload", str(f),
                      f"{REMOTE_DATA}/labels/{split}/{f.name}",
                      "-s", session_name, check=False, timeout=60)

    # Patch data.yaml to use remote paths
    patch_code = f"""
import yaml
from pathlib import Path
cfg = Path("{REMOTE_CONFIGS}/data.yaml")
with open(cfg) as f:
    d = yaml.safe_load(f)
d['path'] = "{REMOTE_DATA}/.."
import yaml
with open(cfg, 'w') as f:
    yaml.dump(d, f)
print("data.yaml patched for remote paths.")
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(patch_code)
        patch_script = Path(tf.name)
    try:
        colab("install", "pyyaml", "-s", session_name, timeout=120)
        colab("exec", "-f", str(patch_script), "-s", session_name, timeout=60)
    finally:
        patch_script.unlink(missing_ok=True)


def verify_gpu(session_name: str) -> bool:
    """Run nvidia-smi on the VM and confirm T4 GPU."""
    check_code = """
import subprocess, sys
result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        capture_output=True, text=True)
gpu_name = result.stdout.strip()
print(f"GPU: {gpu_name}")
if not gpu_name:
    print("ERROR: No GPU detected! Colab provisioned a CPU runtime.", file=sys.stderr)
    sys.exit(1)
if "T4" not in gpu_name and "A100" not in gpu_name and "L4" not in gpu_name:
    print(f"WARNING: Expected T4 but got '{gpu_name}'. Continuing anyway.")
else:
    print(f"✓ GPU confirmed: {gpu_name}")
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(check_code)
        check_script = Path(tf.name)
    try:
        result = colab("exec", "-f", str(check_script), "-s", session_name,
                       check=False, timeout=30)
        return result.returncode == 0
    finally:
        check_script.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Orchestrate YOLO26 training on Colab T4.")
    parser.add_argument("--session", "-s", default="colony_train",
                        help="Colab session name.")
    parser.add_argument("--gpu", default="T4", choices=["T4", "L4", "G4", "A100", "H100"],
                        help="GPU variant to request.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size (-1 = auto).")
    parser.add_argument("--no-drive", action="store_true",
                        help="Skip Google Drive mount (data only on ephemeral VM disk).")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip dataset upload (use if already uploaded to Drive).")
    args = parser.parse_args()

    session = args.session
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    session_started = False

    try:
        # ── Step 1: Provision GPU session ──────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 1: Provisioning Colab T4 session")
        print("=" * 60)
        print(f"\n  >>> If this is your first time using Colab CLI, a browser")
        print(f"  >>> window will open for Google OAuth authentication.")
        print(f"  >>> Complete the login, then return here.")
        input("\n  Press Enter to start provisioning (will open browser for auth if needed) ... ")

        colab("new", "--gpu", args.gpu, "--session", session, timeout=120)
        session_started = True

        # ── Step 2: Verify GPU ─────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 2: Verifying GPU assignment")
        print("=" * 60)
        # Give the VM a moment to fully initialize
        time.sleep(5)
        colab("status", "-s", session, timeout=30)

        if not verify_gpu(session):
            print("\nERROR: GPU verification failed. Stopping.", file=sys.stderr)
            stop_session(session)
            sys.exit(1)
        print("  ✓ GPU verified.")

        # ── Step 3: Mount Google Drive ─────────────────────────────────────────
        if not args.no_drive:
            print("\n" + "=" * 60)
            print("STEP 3: Mounting Google Drive")
            print("=" * 60)
            colab("drivemount", "-s", session, timeout=120)

        # ── Step 4: Upload dataset ─────────────────────────────────────────────
        if not args.no_upload:
            print("\n" + "=" * 60)
            print("STEP 4: Uploading dataset")
            print("=" * 60)
            upload_dataset(session, use_drive=not args.no_drive)

        # ── Step 5: Install Ultralytics ────────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 5: Installing Ultralytics on VM")
        print("=" * 60)
        colab("install", "ultralytics", "-s", session, timeout=300)

        # ── Step 6: Start training (with checkpoint background thread) ─────────
        print("\n" + "=" * 60)
        print("STEP 6: Starting training")
        print("=" * 60)

        remote_last = f"{REMOTE_ROOT}/runs/yolo26n_colony/weights/last.pt"
        local_last = WEIGHTS_DIR / "last.pt"

        stop_checkpoint = threading.Event()
        checkpoint_thread = threading.Thread(
            target=checkpoint_loop,
            args=(session, remote_last, local_last, stop_checkpoint),
            daemon=True
        )
        checkpoint_thread.start()

        # Write training stub and upload it
        stub_script = make_training_stub(args)
        colab("upload", str(stub_script),
              f"{REMOTE_ROOT}/colony_train_stub.py",
              "-s", session, timeout=60)

        # Execute training — long timeout
        print(f"\n  Running training (timeout={TRAINING_TIMEOUT}s / {TRAINING_TIMEOUT//3600}h) …")
        print("  Ultralytics will print GPU memory usage and loss per epoch.")
        print("  Checkpoints are downloaded to weights/last.pt every 30 min.\n")
        colab("exec", "-f", str(stub_script),
              "-s", session,
              "--timeout", str(float(TRAINING_TIMEOUT)),
              timeout=TRAINING_TIMEOUT + 120)

        stop_checkpoint.set()
        stub_script.unlink(missing_ok=True)

        # ── Step 7: Download best weights ─────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 7: Downloading best.pt")
        print("=" * 60)
        remote_best = f"{REMOTE_ROOT}/runs/yolo26n_colony/weights/best.pt"
        colab("download", remote_best, "-s", session, timeout=300)
        # Move to weights/ dir
        downloaded_best = Path("best.pt")
        if downloaded_best.exists():
            import shutil
            shutil.move(str(downloaded_best), str(WEIGHTS_DIR / "best.pt"))
        print(f"  ✓ best.pt saved to {WEIGHTS_DIR / 'best.pt'}")

        # Also download last.pt for resumability
        colab("download", remote_last, "-s", session, check=False, timeout=300)
        downloaded_last = Path("last.pt")
        if downloaded_last.exists():
            import shutil
            shutil.move(str(downloaded_last), str(WEIGHTS_DIR / "last.pt"))

        # ── Step 8: Archive session log ───────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 8: Saving session log")
        print("=" * 60)
        log_path = RUNS_DIR / "colab_session_log.ipynb"
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        colab("log", "-s", session, "-o", str(log_path), check=False, timeout=60)
        if log_path.exists():
            print(f"  ✓ Session log saved to {log_path}")

        print("\n" + "=" * 60)
        print("✓ Colab training pipeline complete!")
        print("=" * 60)
        print(f"\n  best.pt  → {WEIGHTS_DIR / 'best.pt'}")
        print(f"  last.pt  → {WEIGHTS_DIR / 'last.pt'}")
        print(f"  log      → {log_path}")
        print(f"\nNext: run scripts/export_model.py to produce ONNX + OpenVINO exports.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
    finally:
        # ── Step 9: Always stop the session ───────────────────────────────────
        if session_started:
            print("\n" + "=" * 60)
            print("STEP 9 (cleanup): Stopping Colab session")
            print("=" * 60)
            stop_session(session)
            print("  ✓ Session stopped. No lingering billed runtime.")


if __name__ == "__main__":
    main()
