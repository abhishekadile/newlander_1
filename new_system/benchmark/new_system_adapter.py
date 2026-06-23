"""
new_system_adapter.py

Adapter for the YOLO26n-based colony detection system.
Loads the exported model and runs inference, post-processing detections
above confidence threshold conf=0.25.

Model priority: OpenVINO (.xml) → ONNX (.onnx) → PyTorch (.pt)

Returns:
    {"count": int, "latency_ms": float, "detections": list, "error": str | None}
"""

import json
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
CONF_THRESH  = 0.25
IOU_THRESH   = 0.45
IMGSZ        = 640


# ---------------------------------------------------------------------------
# Model loading (priority: OpenVINO → ONNX → PT)
# ---------------------------------------------------------------------------

def _find_model() -> tuple[str, str]:
    """Returns (model_path, backend) where backend is one of 'openvino'|'onnx'|'pt'."""
    # OpenVINO: pass the directory, not the .xml file (ultralytics expects the folder)
    ov_dir = WEIGHTS_DIR / "best_openvino_model"
    if ov_dir.exists() and list(ov_dir.glob("*.xml")):
        return str(ov_dir), "openvino"

    # ONNX
    onnx = WEIGHTS_DIR / "best.onnx"
    if onnx.exists():
        return str(onnx), "onnx"

    # PyTorch
    pt = WEIGHTS_DIR / "best.pt"
    if pt.exists():
        return str(pt), "pt"

    raise FileNotFoundError(
        f"No model weights found in {WEIGHTS_DIR}. "
        "Run Phase 3-4 (training + export) first, then sync weights locally."
    )


_model_cache: object = None
_model_backend: str = ""


def _load_model():
    global _model_cache, _model_backend
    if _model_cache is not None:
        return _model_cache, _model_backend

    model_path, backend = _find_model()
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        _model_cache = model
        _model_backend = backend
        return model, backend
    except ImportError:
        raise ImportError(
            "ultralytics is not installed. "
            "Run: pip install ultralytics"
        )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(image_path: str) -> tuple[list, float]:
    """
    Returns (detections, latency_ms).
    detections: list of {"bbox": [x1,y1,x2,y2], "conf": float, "class_id": int}
    """
    model, backend = _load_model()

    t0 = time.perf_counter()
    results = model.predict(
        source=image_path,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=IMGSZ,
        verbose=False,
        device="cpu",   # benchmark runs locally — CPU inference
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            detections.append({"bbox": xyxy, "conf": conf, "class_id": cls})

    return detections, latency_ms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(image_path: str) -> dict:
    """
    Detect colonies in image_path using the YOLO26n system.

    Returns:
        {
            "count":       int,
            "latency_ms":  float,
            "detections":  list[{"bbox": [...], "conf": float, "class_id": int}],
            "error":       str | None,
            "backend":     str,      # "openvino" | "onnx" | "pt"
        }
    """
    try:
        _, backend = _find_model()
        detections, latency_ms = _run_inference(image_path)
        return {
            "count":       len(detections),
            "latency_ms":  latency_ms,
            "detections":  detections,
            "error":       None,
            "backend":     backend,
        }
    except Exception as exc:
        return {
            "count":       -1,
            "latency_ms":  float("nan"),
            "detections":  [],
            "error":       str(exc),
            "backend":     "unknown",
        }


def warmup(image_path: str) -> None:
    """Run one inference pass to warm up the model (not measured)."""
    try:
        _run_inference(image_path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python new_system_adapter.py <image_path>")
        sys.exit(1)
    result = run(sys.argv[1])
    print(json.dumps({k: v for k, v in result.items() if k != "detections"}, indent=2))
    print(f"Detections: {len(result['detections'])}")
