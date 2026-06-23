"""
sync_checkpoints.py — runs LOCALLY during Colab training.

Periodically downloads best.pt and last.pt from the Colab VM to
new_system/weights/, providing a "last known safe" local copy.

Usage:
    python new_system/scripts/sync_checkpoints.py [--interval 600]

Run in a separate terminal while run_colab_training.sh is active.
Press Ctrl-C to stop.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT    = Path(__file__).parent.parent.parent
WEIGHTS_DIR  = Path(__file__).parent.parent / "weights"
RUNS_DIR     = Path(__file__).parent.parent / "runs"

# Ultralytics prefixes the task type (detect/) to the project path
REMOTE_CHECKPOINTS = [
    ("/content/runs/detect/runs/yolo26n_colony/weights/best.pt",  WEIGHTS_DIR / "best.pt"),
    ("/content/runs/detect/runs/yolo26n_colony/weights/last.pt",  WEIGHTS_DIR / "last.pt"),
]

DEFAULT_INTERVAL = 600   # 10 minutes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[sync {ts()}] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[ALERT {ts()}] {msg}", flush=True)


def run_colab_download(remote: str, local: Path) -> tuple[bool, str]:
    """
    Run: colab download <remote> <local>
    Returns (success, message).
    """
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["colab", "download", remote, str(local)],
            capture_output=True, timeout=120,
        )
        out = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        if result.returncode == 0:
            size_mb = local.stat().st_size / 1024**2 if local.exists() else 0.0
            return True, f"{size_mb:.1f} MB"
        else:
            return False, out.strip()[:200]
    except subprocess.TimeoutExpired:
        return False, "timeout (>120s)"
    except FileNotFoundError:
        return False, "colab CLI not found"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Recovery instructions
# ---------------------------------------------------------------------------

RECOVERY_INSTRUCTIONS = """
Recovery steps after session disconnect:
  1. Provision a new session:
       colab new --gpu T4
  2. Mount Drive (if you used drivemount earlier):
       colab drivemount
  3. Re-upload training scripts:
       colab upload new_system/scripts/train_colony_model.py train_colony_model.py
       colab upload new_system/scripts/export_model.py export_model.py
  4. Upload your last synced checkpoint:
       colab upload new_system/weights/last.pt runs/yolo26n_colony/weights/last.pt
     Note: you may need to create the remote directory first:
       colab exec -c "mkdir -p runs/yolo26n_colony/weights"
  5. Resume training (train_colony_model.py auto-detects last.pt):
       colab exec -c "nohup python train_colony_model.py > train.log 2>&1 &"
  6. Restart this sync script in your terminal.
"""


# ---------------------------------------------------------------------------
# Main sync loop
# ---------------------------------------------------------------------------

def sync_loop(interval: int) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Starting checkpoint sync: interval={interval}s ({interval//60} min)")
    log(f"Local weights directory: {WEIGHTS_DIR}")
    log("Press Ctrl-C to stop.\n")

    consecutive_failures = 0

    while True:
        try:
            any_success = False
            for remote_path, local_path in REMOTE_CHECKPOINTS:
                ok, detail = run_colab_download(remote_path, local_path)
                if ok:
                    log(f"Synced {local_path.name} ({detail}) — last safe checkpoint at {ts()}")
                    any_success = True
                    consecutive_failures = 0
                else:
                    warn(
                        f"Failed to sync {local_path.name}: {detail}"
                    )

            if not any_success:
                consecutive_failures += 1
                warn(
                    f"All checkpoint syncs failed ({consecutive_failures} consecutive). "
                    "Colab session may have disconnected."
                )
                if consecutive_failures >= 2:
                    warn("Session appears to be down. Last successfully synced checkpoints:")
                    for _, local_path in REMOTE_CHECKPOINTS:
                        if local_path.exists():
                            mtime = datetime.fromtimestamp(local_path.stat().st_mtime)
                            size_mb = local_path.stat().st_size / 1024**2
                            warn(f"  {local_path.name}: {size_mb:.1f} MB, "
                                 f"last modified {mtime.strftime('%H:%M:%S')}")
                        else:
                            warn(f"  {local_path.name}: not present locally")
                    print(RECOVERY_INSTRUCTIONS, flush=True)

            time.sleep(interval)

        except KeyboardInterrupt:
            log("Sync stopped by user.")
            log("Local checkpoints retained in:")
            for _, local_path in REMOTE_CHECKPOINTS:
                if local_path.exists():
                    size_mb = local_path.stat().st_size / 1024**2
                    log(f"  {local_path} ({size_mb:.1f} MB)")
            sys.exit(0)
        except Exception as exc:
            warn(f"Unexpected error in sync loop: {exc}")
            time.sleep(interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Periodic best.pt/last.pt sync from Colab VM to local machine."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Sync interval in seconds (default: {DEFAULT_INTERVAL} = 10 min).",
    )
    args = parser.parse_args()
    sync_loop(args.interval)


if __name__ == "__main__":
    main()
