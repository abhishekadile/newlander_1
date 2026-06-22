#!/usr/bin/env python3
"""
train_colony_model.py
=====================
Phase 3: Train YOLO26n on the converted colony dataset.

Features:
  - Auto-detects and uses YOLO26 segmentation (yolo26n-seg.pt) or
    falls back to YOLO26 detection (yolo26n.pt) based on label format.
  - Auto-resumes from last.pt if a previous training run exists.
  - Mixed precision (AMP), auto batch sizing, RAM/disk caching.
  - Post-training validation with recall-driven imgsz retry.
  - Checkpoint every 5 epochs (Colab disconnect protection).

Usage (on Colab T4 — run via run_colab_training.sh):
    python scripts/train_colony_model.py [--resume] [--imgsz 1280] [--epochs 150]

Usage (local testing):
    python scripts/train_colony_model.py --device cpu --epochs 5 --batch 8
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# ── Paths (resolved relative to this script's location) ───────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent       # new_system/
DATA_YAML = ROOT / "configs" / "data.yaml"
RUNS_DIR = ROOT / "runs"
WEIGHTS_DIR = ROOT / "weights"
LAST_PT = RUNS_DIR / "yolo26n_colony" / "weights" / "last.pt"
BEST_PT = RUNS_DIR / "yolo26n_colony" / "weights" / "best.pt"

# ── Model selection ────────────────────────────────────────────────────────────
SEG_MODEL = "yolo26n-seg.pt"
DET_MODEL = "yolo26n.pt"


def verify_ultralytics():
    """Ensure ultralytics is installed and YOLO26 is importable."""
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("ultralytics not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics"])
        from ultralytics import YOLO
        return YOLO


def detect_label_format() -> str:
    """Return 'seg' if labels contain polygon annotations, else 'det'."""
    labels_dir = ROOT / "data" / "processed" / "labels" / "train"
    if not labels_dir.exists():
        return "det"
    # Sample up to 20 label files
    samples = list(labels_dir.glob("*.txt"))[:20]
    for lbl in samples:
        text = lbl.read_text(encoding="utf-8").strip()
        if not text:
            continue
        line = text.splitlines()[0].strip().split()
        # YOLO seg lines have >5 fields (class + ≥4 xy pairs = 9+)
        # YOLO det lines have exactly 5 fields
        if len(line) > 5:
            return "seg"
    return "det"


def pick_model(YOLO, label_fmt: str) -> tuple:
    """
    Try seg model first. Fall back to det model if seg checkpoint unavailable.
    Returns (model_name, YOLO_instance).
    """
    if label_fmt == "seg":
        try:
            model = YOLO(SEG_MODEL)
            print(f"  Using segmentation model: {SEG_MODEL}")
            return SEG_MODEL, model
        except Exception as e:
            print(f"  WARNING: {SEG_MODEL} not available ({e}). Falling back to {DET_MODEL}.")
    model = YOLO(DET_MODEL)
    print(f"  Using detection model: {DET_MODEL}")
    return DET_MODEL, model


def train(args):
    print("\n=== YOLO26 Colony Training ===")
    print(f"  data.yaml:  {DATA_YAML}")
    print(f"  runs dir:   {RUNS_DIR}")
    print(f"  epochs:     {args.epochs}")
    print(f"  imgsz:      {args.imgsz}")
    print(f"  device:     {args.device}")
    print(f"  batch:      {args.batch}")
    print()

    # Validate data.yaml exists
    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found. Run convert_to_yolo.py first.", file=sys.stderr)
        sys.exit(1)

    YOLO = verify_ultralytics()
    label_fmt = detect_label_format()
    print(f"  Detected label format: {label_fmt}")

    # ── Auto-resume logic ──────────────────────────────────────────────────────
    resume = args.resume
    model_name = None
    model = None

    if LAST_PT.exists() and not args.no_resume:
        print(f"\n  Found existing checkpoint: {LAST_PT}")
        print("  Resuming training (resume=True) ...")
        model = YOLO(str(LAST_PT))
        model_name = str(LAST_PT)
        resume = True
    else:
        model_name, model = pick_model(YOLO, label_fmt)
        resume = False

    # ── Cache strategy ─────────────────────────────────────────────────────────
    # Estimate dataset size to decide RAM vs disk caching
    images_train = ROOT / "data" / "processed" / "images" / "train"
    total_bytes = sum(f.stat().st_size for f in images_train.rglob("*") if f.is_file()) if images_train.exists() else 0
    total_gb = total_bytes / (1024 ** 3)
    cache_mode = "ram" if total_gb < 8 else "disk"
    print(f"  Dataset size: ~{total_gb:.2f} GB → cache='{cache_mode}'")

    # ── Training ───────────────────────────────────────────────────────────────
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    train_kwargs = dict(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        amp=True,
        cache=cache_mode,
        workers=args.workers,
        patience=25,
        project=str(RUNS_DIR),
        name="yolo26n_colony",
        save_period=5,
        resume=resume,
        exist_ok=True,
    )

    print(f"\n  Training parameters:")
    for k, v in train_kwargs.items():
        print(f"    {k}: {v}")
    print()

    results = model.train(**train_kwargs)

    # ── Post-training validation ────────────────────────────────────────────────
    print("\n=== Post-Training Validation ===")
    val_results = model.val(data=str(DATA_YAML))

    recall = None
    try:
        # Ultralytics stores recall in results.results_dict
        r = val_results.results_dict
        recall = r.get("metrics/recall(B)", r.get("recall", None))
        if recall is not None:
            print(f"  Val recall: {recall:.4f}")
    except Exception:
        pass

    # ── Retry at imgsz=1280 if recall is poor ──────────────────────────────────
    if args.imgsz < 1280 and recall is not None and recall < 0.6 and not args.no_retry:
        print(f"\n  Val recall {recall:.4f} < 0.60 — retrying at imgsz=1280 …")
        model.train(
            **{**train_kwargs, "imgsz": 1280, "name": "yolo26n_colony_1280", "resume": False}
        )
        val_results = model.val(data=str(DATA_YAML))

    # ── Copy best weights to weights/ ──────────────────────────────────────────
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if BEST_PT.exists():
        import shutil
        shutil.copy2(BEST_PT, WEIGHTS_DIR / "best.pt")
        print(f"\n  Copied best.pt to {WEIGHTS_DIR / 'best.pt'}")
    if LAST_PT.exists():
        import shutil
        shutil.copy2(LAST_PT, WEIGHTS_DIR / "last.pt")
        print(f"  Copied last.pt to {WEIGHTS_DIR / 'last.pt'}")

    print("\n✓ Training complete.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26n on colony dataset.")
    parser.add_argument("--resume", action="store_true",
                        help="Force resume=True (normally auto-detected from last.pt).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Force fresh training even if last.pt exists.")
    parser.add_argument("--no-retry", action="store_true",
                        help="Skip the imgsz=1280 retry if recall is low.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size. -1 = auto (recommended for Colab T4).")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
