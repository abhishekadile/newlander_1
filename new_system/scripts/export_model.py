#!/usr/bin/env python3
"""
export_model.py
===============
Phase 5: Export trained best.pt weights to ONNX and OpenVINO formats.

The exported models are saved into new_system/weights/.

Usage:
    cd new_system/
    python scripts/export_model.py [--weights weights/best.pt] [--imgsz 640]

Requirements:
    pip install ultralytics openvino
"""

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
WEIGHTS_DIR = ROOT / "weights"


def main():
    parser = argparse.ArgumentParser(description="Export YOLO26 model to ONNX and OpenVINO.")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_DIR / "best.pt"),
                        help="Path to trained .pt weights file.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true",
                        help="Export FP16 weights (for compatible hardware).")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: weights not found at {weights_path}", file=sys.stderr)
        print(f"  Run run_colab_training.sh to train and download best.pt first.", file=sys.stderr)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Exporting {weights_path.name} ===")
    print(f"  imgsz: {args.imgsz}")
    print(f"  half (FP16): {args.half}")
    print()

    model = YOLO(str(weights_path))

    # ── ONNX export ────────────────────────────────────────────────────────────
    print("  Exporting to ONNX …")
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        dynamic=False,
        simplify=True,
        opset=17,
    )
    # Copy to weights/ if not already there
    if onnx_path and Path(onnx_path).exists():
        dest = WEIGHTS_DIR / Path(onnx_path).name
        if Path(onnx_path).resolve() != dest.resolve():
            shutil.copy2(onnx_path, dest)
        print(f"  ✓ ONNX: {dest}")
    else:
        print(f"  WARNING: ONNX export returned path {onnx_path}", file=sys.stderr)

    # ── OpenVINO export ────────────────────────────────────────────────────────
    print("\n  Exporting to OpenVINO …")
    ov_path = model.export(
        format="openvino",
        imgsz=args.imgsz,
        half=args.half,
        dynamic=False,
    )
    # OpenVINO export produces a directory (model_openvino/)
    if ov_path and Path(ov_path).exists():
        ov_src = Path(ov_path)
        ov_dest = WEIGHTS_DIR / ov_src.name
        if ov_src.resolve() != ov_dest.resolve():
            if ov_dest.exists():
                shutil.rmtree(ov_dest)
            shutil.copytree(ov_src, ov_dest)
        print(f"  ✓ OpenVINO: {ov_dest}")
    else:
        print(f"  WARNING: OpenVINO export returned path {ov_path}", file=sys.stderr)

    print("\n✓ Export complete.")
    print(f"\nWeights directory contents:")
    for p in sorted(WEIGHTS_DIR.rglob("*"))[:30]:
        rel = p.relative_to(WEIGHTS_DIR)
        size = f"{p.stat().st_size / 1024:.1f} KB" if p.is_file() else "[dir]"
        print(f"  {rel}  {size}")


if __name__ == "__main__":
    main()
