"""
export_model.py — runs ON the Colab VM, NOT locally.

Executed via: colab exec -f scripts/export_model.py
  (called from run_colab_training.sh after training completes)

Exports the trained best.pt to:
  - ONNX  (.onnx)
  - OpenVINO (.xml + .bin)

Both export outputs are placed alongside best.pt in runs/yolo26n_colony/weights/.
Download them locally with:
    colab download runs/yolo26n_colony/weights/best.onnx new_system/weights/best.onnx
    colab download runs/yolo26n_colony/weights/best_openvino_model new_system/weights/best_openvino_model
"""

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BEST_PT   = Path("runs/yolo26n_colony/weights/best.pt")
IMGSZ     = 640


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_model() -> None:
    if not BEST_PT.exists():
        die(
            f"best.pt not found at {BEST_PT}. "
            "Ensure training completed and weights were saved."
        )

    log(f"Loading {BEST_PT} …")
    try:
        from ultralytics import YOLO
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
        from ultralytics import YOLO

    model = YOLO(str(BEST_PT))

    # --- ONNX export ---
    log(f"Exporting to ONNX (imgsz={IMGSZ}) …")
    try:
        onnx_path = model.export(format="onnx", imgsz=IMGSZ)
        log(f"ONNX export: {onnx_path}")
    except Exception as exc:
        log(f"WARNING: ONNX export failed: {exc}")
        log("Continuing to OpenVINO export.")

    # --- OpenVINO export ---
    log(f"Exporting to OpenVINO (imgsz={IMGSZ}) …")
    try:
        ov_path = model.export(format="openvino", imgsz=IMGSZ)
        log(f"OpenVINO export: {ov_path}")
    except Exception as exc:
        log(f"WARNING: OpenVINO export failed: {exc}")
        log(
            "If openvino-dev is missing, install it: "
            "pip install openvino-dev ultralytics"
        )

    # --- Summary ---
    weights_dir = BEST_PT.parent
    log("")
    log("Export complete. Files in weights directory:")
    for f in sorted(weights_dir.iterdir()):
        if f.is_file():
            log(f"  {f.name}  ({f.stat().st_size / 1024**2:.1f} MB)")
        elif f.is_dir():
            log(f"  {f.name}/  (directory)")

    log("")
    log("Download commands to run locally:")
    log(f"  colab download {weights_dir}/best.onnx new_system/weights/best.onnx")
    log(f"  colab download {weights_dir}/best_openvino_model new_system/weights/best_openvino_model")


if __name__ == "__main__":
    export_model()
