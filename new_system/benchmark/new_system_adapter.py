#!/usr/bin/env python3
"""
new_system_adapter.py
=====================
Benchmark adapter for the new YOLO26 + OpenVINO colony detection system.

Loads the exported OpenVINO model from new_system/weights/ and runs inference
on the same images as current_system_adapter.py.

Returns:
    {
        "count": int,
        "colonies": list[dict],   # bounding box / polygon coords per colony
        "masks": list or None,    # segmentation masks if seg model
        "latency_ms": float,
    }
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).resolve().parent
NEW_SYSTEM_ROOT = BENCHMARK_DIR.parent
WEIGHTS_DIR = NEW_SYSTEM_ROOT / "weights"


def _find_openvino_model() -> Optional[Path]:
    """Locate the OpenVINO model directory in weights/."""
    for candidate in WEIGHTS_DIR.glob("*openvino*"):
        if candidate.is_dir():
            # Check for .xml file inside
            xmls = list(candidate.glob("*.xml"))
            if xmls:
                return candidate
    return None


def _find_onnx_model() -> Optional[Path]:
    """Locate an ONNX model file in weights/."""
    for candidate in WEIGHTS_DIR.glob("*.onnx"):
        return candidate
    return None


def _find_pt_model() -> Optional[Path]:
    """Locate a .pt model file in weights/."""
    for name in ("best.pt", "last.pt"):
        p = WEIGHTS_DIR / name
        if p.exists():
            return p
    return None


class YOLOAdapter:
    """
    Wraps the Ultralytics YOLO model (OpenVINO, ONNX, or PT) for inference.
    Selects OpenVINO > ONNX > PT in preference order (deployment targets CPU/VPU).
    """

    def __init__(self):
        self.model = None
        self.model_path = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("pip install ultralytics")

        # Priority: OpenVINO → ONNX → PT
        ov_dir = _find_openvino_model()
        if ov_dir:
            print(f"  [new_system] Loading OpenVINO model: {ov_dir}")
            self.model = YOLO(str(ov_dir))
            self.model_path = ov_dir
            return

        onnx = _find_onnx_model()
        if onnx:
            print(f"  [new_system] Loading ONNX model: {onnx}")
            self.model = YOLO(str(onnx))
            self.model_path = onnx
            return

        pt = _find_pt_model()
        if pt:
            print(f"  [new_system] Loading PT model: {pt} (for benchmark; prefer exporting first)")
            self.model = YOLO(str(pt))
            self.model_path = pt
            return

        raise FileNotFoundError(
            f"No model found in {WEIGHTS_DIR}. "
            "Run run_colab_training.py then export_model.py first."
        )

    def predict(self, image_path: Path) -> dict:
        """
        Run YOLO inference on a single image.
        Returns count, per-colony detections, and raw Ultralytics results.
        """
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size

        results = self.model(str(image_path), verbose=False)

        colonies = []
        masks_out = None
        count = 0

        for result in results:
            # Bounding boxes
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                count = len(boxes)
                for i, box in enumerate(boxes):
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    colony_info = {
                        "x": (xyxy[0] + xyxy[2]) / 2,
                        "y": (xyxy[1] + xyxy[3]) / 2,
                        "width": xyxy[2] - xyxy[0],
                        "height": xyxy[3] - xyxy[1],
                        "confidence": conf,
                        "x1": xyxy[0], "y1": xyxy[1],
                        "x2": xyxy[2], "y2": xyxy[3],
                    }
                    # radius approximation from area
                    import math
                    area = colony_info["width"] * colony_info["height"]
                    colony_info["radius"] = math.sqrt(area / math.pi)
                    colonies.append(colony_info)

            # Segmentation masks (if seg model)
            if result.masks is not None:
                masks_out = []
                for mask in result.masks.xy:
                    masks_out.append(mask.tolist())

        return {
            "count": count,
            "colonies": colonies,
            "masks": masks_out,
        }


# Global model singleton (load once per benchmark run)
_adapter_instance: Optional[YOLOAdapter] = None


def _get_adapter() -> YOLOAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = YOLOAdapter()
    return _adapter_instance


def run(image_path: Path) -> dict:
    """
    Run the new YOLO26 system on image_path.
    Returns count, colonies, masks, and latency_ms.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    adapter = _get_adapter()

    t0 = time.perf_counter()
    result = adapter.predict(image_path)
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "count": result["count"],
        "colonies": result["colonies"],
        "masks": result["masks"],
        "latency_ms": latency_ms,
    }


# ── CLI test entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to test image")
    args = parser.parse_args()
    result = run(Path(args.image))
    result_display = {k: v for k, v in result.items() if k != "masks"}
    print(json.dumps(result_display, indent=2))
