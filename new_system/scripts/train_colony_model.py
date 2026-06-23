"""
train_colony_model.py — runs ON the Colab VM, NOT locally.

Executed via: colab exec -f scripts/train_colony_model.py
  (or as a detached background process during Phase 3 orchestration)

Responsibilities:
  1. Detect whether a previous run exists (auto-resume).
  2. Verify YOLO26n availability; fall back gracefully.
  3. Train with augmentation tuned for small-scene datasets.
  4. Post-training validation on val and test splits.
  5. If small-colony recall is poor on val, retry with imgsz=1280.
  6. Write KNOWN_LIMITATIONS.md to the run directory.

Dataset: Makrai et al. 2023 (CC BY 4.0) — prepared by remote_dataset_setup.py
Model: YOLO26n detection (bbox-only — Makrai annotations are bounding boxes, not masks)
"""

import subprocess
import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def pip_install(*packages: str) -> None:
    log(f"Installing: {' '.join(packages)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def check_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            log(f"GPU: {name} — {mem_gb:.1f} GB VRAM")
        else:
            log("WARNING: No CUDA GPU detected. Training will be very slow on CPU.")
    except ImportError:
        log("torch not yet installed; GPU check deferred.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_YAML       = Path("/content/configs/data.yaml")
RUN_PROJECT     = "runs"
RUN_NAME        = "yolo26n_colony"
RUN_DIR         = Path(RUN_PROJECT) / RUN_NAME
WEIGHTS_DIR     = RUN_DIR / "weights"
LAST_PT         = WEIGHTS_DIR / "last.pt"
BEST_PT         = WEIGHTS_DIR / "best.pt"

MODEL_PREF_SEG  = "yolo26n-seg.pt"   # preferred (segmentation) — only useful with mask anns
MODEL_PREF_DET  = "yolo26n.pt"       # fallback (detection)  ← expected to be used here
MODEL_FALLBACK  = "yolo11n.pt"       # last resort if yolo26n is not yet on ultralytics

TRAIN_KWARGS = dict(
    epochs     = 200,
    imgsz      = 640,
    batch      = -1,          # auto-size to ~60-80 % of T4 VRAM
    device     = 0,
    amp        = True,
    cache      = "ram",
    workers    = 8,
    patience   = 30,          # higher patience — small dataset, don't stop too early
    hsv_v      = 0.6,         # brightness/value jitter — partial proxy for lighting diversity
    mixup      = 0.15,
    mosaic     = 1.0,
    project    = RUN_PROJECT,
    name       = RUN_NAME,
    save_period = 1,           # checkpoint every epoch for frequent sync
)

# Recall threshold below which we retry with imgsz=1280
RECALL_RETRY_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def resolve_model_name() -> str:
    """
    Returns the model name string to pass to YOLO().
    Prefers yolo26n.pt (detection, since Makrai annotations are bbox-only).
    Tries the seg variant first only to log availability; doesn't use it unless
    mask annotations exist.
    """
    from ultralytics import YOLO

    # We know we have bbox annotations only — detection model is correct.
    # Test seg availability purely for informational logging.
    try:
        import ultralytics
        ver = ultralytics.__version__
        log(f"ultralytics version: {ver}")
    except Exception:
        pass

    for model_name in (MODEL_PREF_DET, MODEL_FALLBACK):
        try:
            log(f"Checking model availability: {model_name}")
            YOLO(model_name)   # downloads if not cached
            log(f"Using model: {model_name}")
            return model_name
        except Exception as exc:
            log(f"  {model_name} unavailable: {exc}")

    raise RuntimeError("No suitable YOLO26 model found. "
                       "Ensure ultralytics is up to date: pip install -U ultralytics")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(model_name: str, resume: bool, extra_kwargs: dict | None = None) -> object:
    from ultralytics import YOLO

    kwargs = dict(TRAIN_KWARGS)
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {DATA_YAML}. "
            "Run remote_dataset_setup.py first."
        )

    if resume:
        log(f"Resuming from {LAST_PT}")
        model = YOLO(str(LAST_PT))
        results = model.train(data=str(DATA_YAML), resume=True, **kwargs)
    else:
        log(f"Starting fresh training: {model_name}")
        model = YOLO(model_name)
        results = model.train(data=str(DATA_YAML), resume=False, **kwargs)

    return results


# ---------------------------------------------------------------------------
# Post-training validation
# ---------------------------------------------------------------------------

def validate(split: str = "val", weights: Path = BEST_PT) -> dict:
    from ultralytics import YOLO
    log(f"Validating on '{split}' split using {weights} …")
    model = YOLO(str(weights))
    metrics = model.val(data=str(DATA_YAML), split=split, device=0)
    recall = metrics.box.r.mean() if hasattr(metrics, "box") else float("nan")
    map50  = metrics.box.map50    if hasattr(metrics, "box") else float("nan")
    map5095 = metrics.box.map     if hasattr(metrics, "box") else float("nan")
    log(f"  {split}: recall={recall:.3f}, mAP50={map50:.3f}, mAP50-95={map5095:.3f}")
    return {"recall": recall, "map50": map50, "map5095": map5095, "split": split}


# ---------------------------------------------------------------------------
# Known-limitations file
# ---------------------------------------------------------------------------

KNOWN_LIMITATIONS_TEXT = textwrap.dedent("""\
    # Known Limitations — yolo26n_colony Training Run

    ## 1. Merged / touching colonies — NOT EVALUATED

    The MCount dataset (Dryad), which contains merged and touching colony images,
    is currently inaccessible (locked as of June 2026). This model has NOT been
    trained or validated on merged/touching colony scenarios.

    **Action required:** Revisit once MCount access is restored, or once
    enough real-world touching/merged-colony images are collected from production
    IncuCountAPI deployments.

    ## 2. Glare and variable lighting — NOT FULLY EVALUATED

    Makrai et al. 2023 provides images with two background conditions: white and
    black agar plates. Real-world glare, outdoor lighting, and other illumination
    variation beyond this binary condition have NOT been confirmed in training data.

    The `hsv_v=0.6` augmentation provides a partial proxy for brightness/value
    variation, but it is not a substitute for genuine lighting-diversity data.

    ## 3. Annotation type — bounding boxes only

    Makrai annotations are axis-aligned bounding boxes. Segmentation masks were
    NOT generated or used. The YOLO26n (detection) model is appropriate for this
    annotation type, but instance segmentation accuracy cannot be claimed.

    ## 4. Dataset scale

    ~369 scene images (56,865 annotated colonies, 24 species).
    Scene diversity is limited relative to instance count. The augmentation
    strategy (mosaic, mixup, hsv_v) partially compensates, but is not a
    substitute for more scene variety.

    ---
    This file was auto-generated by train_colony_model.py.
""")


def write_known_limitations() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out = RUN_DIR / "KNOWN_LIMITATIONS.md"
    out.write_text(KNOWN_LIMITATIONS_TEXT)
    log(f"KNOWN_LIMITATIONS.md written to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=== YOLO26n Colony Detection — Training (remote, on Colab VM) ===")

    # Install dependencies if not present
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        pip_install("ultralytics")

    check_gpu()

    # Auto-resume detection
    resume = LAST_PT.exists()
    if resume:
        log(f"Detected existing checkpoint at {LAST_PT} — will resume.")
    else:
        log("No existing checkpoint found — starting fresh training.")

    # Model selection (only needed for fresh start)
    model_name = resolve_model_name() if not resume else str(LAST_PT)

    # --- First training pass (imgsz=640) ---
    log("")
    log("--- Training pass 1 / 2: imgsz=640 ---")
    run_training(model_name, resume=resume)

    # --- Post-training validation ---
    log("")
    log("--- Post-training validation ---")
    val_metrics  = validate("val",  BEST_PT)
    test_metrics = validate("test", BEST_PT)

    # --- imgsz=1280 retry if recall is poor ---
    if val_metrics["recall"] < RECALL_RETRY_THRESHOLD:
        log("")
        log(
            f"Val recall {val_metrics['recall']:.3f} < {RECALL_RETRY_THRESHOLD} "
            f"— retrying with imgsz=1280 (one retry pass)."
        )
        run_training(str(BEST_PT), resume=False, extra_kwargs={"imgsz": 1280, "name": f"{RUN_NAME}_1280"})
        log("")
        log("--- Validation after imgsz=1280 retry ---")
        retry_dir = Path(RUN_PROJECT) / f"{RUN_NAME}_1280"
        retry_best = retry_dir / "weights" / "best.pt"
        if retry_best.exists():
            val_metrics_retry  = validate("val",  retry_best)
            test_metrics_retry = validate("test", retry_best)
            log(f"imgsz=1280 val recall: {val_metrics_retry['recall']:.3f}")
        else:
            log("WARNING: imgsz=1280 best.pt not found — using original best.pt.")
    else:
        log(f"Val recall {val_metrics['recall']:.3f} is acceptable — no imgsz retry needed.")

    # --- Summary ---
    log("")
    log("=== Training Complete ===")
    log(f"  best.pt  : {BEST_PT}")
    log(f"  last.pt  : {LAST_PT}")
    log(f"  val  recall={val_metrics['recall']:.3f}  mAP50={val_metrics['map50']:.3f}")
    log(f"  test recall={test_metrics['recall']:.3f}  mAP50={test_metrics['map50']:.3f}")

    # --- Known limitations ---
    write_known_limitations()

    # Final reminder about known limitations
    log("")
    log("KNOWN LIMITATION: This model has NOT been trained or validated on")
    log("merged/touching colonies (no MCount data available this pass) and has")
    log("NOT been validated for real-world glare/lighting beyond white/black")
    log("background variation. See KNOWN_LIMITATIONS.md for details.")
    log("")
    log("Sync best.pt to local machine with:")
    log("  colab download runs/yolo26n_colony/weights/best.pt new_system/weights/best.pt")


if __name__ == "__main__":
    main()
