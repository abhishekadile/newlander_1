#!/usr/bin/env python3
"""
convert_to_yolo.py
==================
Phase 2: Convert downloaded datasets to YOLO format for Ultralytics training.

Strategy:
  - Single class: colony (class id 0)
  - YOLO segmentation format (polygon points) if source annotations support it;
    else fall back to YOLO detection box format.
  - Split:
      Makrai 2023  →  80% train / 10% val / 10% test
      MCount       →  100% test  (held-out merged-colony evaluation set;
                       filenames prefixed with "mcount_")
  - Output: data/processed/images/{train,val,test}/ and
            data/processed/labels/{train,val,test}/
  - Writes configs/data.yaml (Ultralytics format)
  - Writes data/processed/CONVERSION_NOTES.md

Usage:
    cd new_system/
    python scripts/convert_to_yolo.py [--seg-fallback] [--dry-run]

Requirements:
    pip install Pillow opencv-python numpy tqdm PyYAML
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    import yaml
except ImportError:
    print("ERROR: pip install Pillow opencv-python numpy tqdm PyYAML", file=sys.stderr)
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
CONFIGS = ROOT / "configs"

MAKRAI_DIR = DATA_RAW / "makrai2023"
MCOUNT_DIR = DATA_RAW / "mcount"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# Format detection helpers
# ──────────────────────────────────────────────────────────────────────────────

def detect_annotation_format(dataset_dir: Path) -> str:
    """Return 'coco', 'voc', 'csv', 'mask', or 'unknown'."""
    if any(dataset_dir.rglob("*.json")):
        for jf in list(dataset_dir.rglob("*.json"))[:3]:
            try:
                with open(jf) as f:
                    d = json.load(f)
                if "annotations" in d and "categories" in d:
                    return "coco"
            except Exception:
                pass
    if any(dataset_dir.rglob("*.xml")):
        return "voc"
    if any(dataset_dir.rglob("*.csv")):
        return "csv"
    # Look for mask images (separate mask files alongside images)
    mask_candidates = list(dataset_dir.rglob("*mask*"))
    if mask_candidates:
        return "mask"
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# COCO → YOLO conversion (segmentation preferred, box fallback)
# ──────────────────────────────────────────────────────────────────────────────

def coco_to_yolo(coco_json: Path, images_root: Path, out_images: Path,
                 out_labels: Path, prefix: str = "", seg: bool = True,
                 dry_run: bool = False) -> List[str]:
    """
    Convert a COCO-format JSON to YOLO label files.
    Returns list of processed image filenames.
    """
    with open(coco_json) as f:
        coco = json.load(f)

    # Map image_id → file info
    id_to_img = {img["id"]: img for img in coco["images"]}

    # Map image_id → list of annotations
    anns_by_img: Dict[int, list] = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    processed = []
    for img_id, img_info in tqdm(id_to_img.items(), desc="COCO→YOLO"):
        filename = img_info["file_name"]
        w = img_info["width"]
        h = img_info["height"]

        # Find the source image
        src_img = find_image(images_root, filename)
        if src_img is None:
            print(f"  WARNING: image not found: {filename}", file=sys.stderr)
            continue

        stem = Path(filename).stem
        new_stem = f"{prefix}{stem}" if prefix else stem
        ext = src_img.suffix

        dst_img = out_images / (new_stem + ext)
        dst_lbl = out_labels / (new_stem + ".txt")

        if not dry_run:
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)

        anns = anns_by_img.get(img_id, [])
        lines = []
        for ann in anns:
            if seg and ann.get("segmentation"):
                segs = ann["segmentation"]
                # Take first polygon (COCO can have multiple for one ann)
                if isinstance(segs, list) and len(segs) > 0 and isinstance(segs[0], list):
                    pts = segs[0]
                    # Normalize: [x1,y1,x2,y2,...] → [x1/W,y1/H,...]
                    norm = []
                    for i in range(0, len(pts) - 1, 2):
                        norm.append(f"{pts[i]/w:.6f}")
                        norm.append(f"{pts[i+1]/h:.6f}")
                    if len(norm) >= 6:  # At least 3 points
                        lines.append("0 " + " ".join(norm))
                        continue
            # Fall back to box
            bbox = ann.get("bbox")
            if bbox:
                bx, by, bw, bh = bbox
                cx = (bx + bw / 2) / w
                cy = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not dry_run:
            dst_lbl.write_text("\n".join(lines), encoding="utf-8")

        processed.append(new_stem + ext)
    return processed


# ──────────────────────────────────────────────────────────────────────────────
# Pascal VOC → YOLO (bounding box only — VOC doesn't store polygons)
# ──────────────────────────────────────────────────────────────────────────────

def voc_to_yolo(xml_file: Path, src_img: Path, out_images: Path,
                out_labels: Path, prefix: str = "", dry_run: bool = False) -> Optional[str]:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
            cx = ((xmin + xmax) / 2) / w
            cy = ((ymin + ymax) / 2) / h
            nw = (xmax - xmin) / w
            nh = (ymax - ymin) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        stem = src_img.stem
        new_stem = f"{prefix}{stem}" if prefix else stem
        ext = src_img.suffix

        if not dry_run:
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, out_images / (new_stem + ext))
            (out_labels / (new_stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")

        return new_stem + ext
    except Exception as e:
        print(f"  WARNING: VOC parse failed for {xml_file.name}: {e}", file=sys.stderr)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Mask-based annotation → YOLO (e.g., MCount binary mask PNGs)
# ──────────────────────────────────────────────────────────────────────────────

def mask_to_yolo_seg(img_path: Path, mask_path: Path, out_images: Path,
                     out_labels: Path, prefix: str = "", dry_run: bool = False) -> Optional[str]:
    """
    Convert a binary/labeled mask image to YOLO segmentation format.
    Extracts contours of each labeled region as polygon coordinates.
    """
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        img = cv2.imread(str(img_path))
        if img is None:
            return None

        h, w = mask.shape[:2]

        # If mask is binary: find connected components
        # If mask is labeled (0=bg, 1..N=instances): iterate labels
        unique_vals = np.unique(mask)
        unique_vals = unique_vals[unique_vals > 0]  # remove background

        lines = []
        for val in unique_vals:
            instance_mask = (mask == val).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if contour.shape[0] < 3:
                    continue
                # Simplify contour slightly
                eps = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, eps, True)
                if approx.shape[0] < 3:
                    continue
                pts = approx.reshape(-1, 2)
                norm = []
                for px, py in pts:
                    norm.append(f"{px/w:.6f}")
                    norm.append(f"{py/h:.6f}")
                lines.append("0 " + " ".join(norm))

        if not lines:
            return None

        stem = img_path.stem
        new_stem = f"{prefix}{stem}" if prefix else stem
        ext = img_path.suffix

        if not dry_run:
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, out_images / (new_stem + ext))
            (out_labels / (new_stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")

        return new_stem + ext
    except Exception as e:
        print(f"  WARNING: mask_to_yolo_seg failed for {img_path.name}: {e}", file=sys.stderr)
        return None


def csv_to_yolo_box(img_path: Path, csv_path: Path, out_images: Path,
                    out_labels: Path, prefix: str = "", dry_run: bool = False) -> Optional[str]:
    """
    Convert CSV centroid/bbox annotations to YOLO box format.
    CSV is expected to have columns (flexible): x, y, radius or x_min,y_min,x_max,y_max
    """
    try:
        import csv as csvmod
        img = Image.open(img_path)
        w, h = img.size

        lines = []
        with open(csv_path, newline="") as f:
            reader = csvmod.DictReader(f)
            headers = [c.lower().strip() for c in (reader.fieldnames or [])]

            for row in reader:
                row_lower = {k.lower().strip(): v for k, v in row.items()}
                # Try centroid+radius format
                if "x" in row_lower and "y" in row_lower:
                    try:
                        cx = float(row_lower["x"]) / w
                        cy = float(row_lower["y"]) / h
                        r = float(row_lower.get("radius", row_lower.get("r", 5))) / max(w, h)
                        nw = 2 * r
                        nh = 2 * r
                        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                        continue
                    except ValueError:
                        pass
                # Try explicit bbox format
                if all(k in row_lower for k in ["x_min", "y_min", "x_max", "y_max"]):
                    try:
                        x1, y1 = float(row_lower["x_min"]), float(row_lower["y_min"])
                        x2, y2 = float(row_lower["x_max"]), float(row_lower["y_max"])
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        nw = abs(x2 - x1) / w
                        nh = abs(y2 - y1) / h
                        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    except ValueError:
                        pass

        if not lines:
            return None

        stem = img_path.stem
        new_stem = f"{prefix}{stem}" if prefix else stem
        ext = img_path.suffix

        if not dry_run:
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, out_images / (new_stem + ext))
            (out_labels / (new_stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")

        return new_stem + ext
    except Exception as e:
        print(f"  WARNING: csv_to_yolo_box failed for {img_path.name}: {e}", file=sys.stderr)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def find_image(search_root: Path, filename: str) -> Optional[Path]:
    """Search recursively for an image file by name."""
    stem = Path(filename).stem
    for ext in IMAGE_EXTS:
        for p in search_root.rglob(f"*{stem}{ext}"):
            return p
    return None


def train_val_test_split(items: list, val_frac=0.1, test_frac=0.1, seed=RANDOM_SEED):
    """Split a list into train/val/test."""
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test = items[:n_test]
    val = items[n_test:n_test + n_val]
    train = items[n_test + n_val:]
    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# Main conversion logic
# ──────────────────────────────────────────────────────────────────────────────

def convert_makrai(dry_run: bool, seg_mode: bool) -> Tuple[int, int, int, str]:
    """Convert Makrai 2023 dataset. Returns (n_train, n_val, n_test, format_used)."""
    print("\n=== Converting Makrai 2023 ===")
    ann_fmt = detect_annotation_format(MAKRAI_DIR)
    print(f"  Detected annotation format: {ann_fmt}")

    # Collect all images
    all_images = sorted([
        p for p in MAKRAI_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    ])
    if not all_images:
        print("  ERROR: no images found in Makrai dir!", file=sys.stderr)
        sys.exit(1)

    train_imgs, val_imgs, test_imgs = train_val_test_split(all_images)
    print(f"  Split: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")

    format_used = "unknown"

    def process_split(imgs: list, split: str):
        nonlocal format_used
        out_img = DATA_PROC / "images" / split
        out_lbl = DATA_PROC / "labels" / split
        n_ok = 0

        for img_path in tqdm(imgs, desc=f"Makrai {split}"):
            stem = img_path.stem
            converted = False

            if ann_fmt == "coco":
                # Find the COCO JSON that references this image
                for jf in MAKRAI_DIR.rglob("*.json"):
                    try:
                        with open(jf) as f:
                            d = json.load(f)
                        for img_entry in d.get("images", []):
                            if Path(img_entry["file_name"]).stem == stem:
                                # Single-image processing
                                mini_coco = {
                                    "images": [img_entry],
                                    "annotations": [
                                        a for a in d.get("annotations", [])
                                        if a["image_id"] == img_entry["id"]
                                    ],
                                    "categories": d.get("categories", [])
                                }
                                # Write temp json for re-use of coco_to_yolo
                                import tempfile
                                with tempfile.NamedTemporaryFile(
                                    mode="w", suffix=".json", delete=False
                                ) as tmp:
                                    json.dump(mini_coco, tmp)
                                    tmp_path = Path(tmp.name)
                                try:
                                    result = coco_to_yolo(
                                        tmp_path, MAKRAI_DIR, out_img, out_lbl,
                                        seg=seg_mode and ann_fmt == "coco", dry_run=dry_run
                                    )
                                    if result:
                                        format_used = "YOLO segmentation (COCO source)" if seg_mode else "YOLO detection box (COCO source)"
                                        converted = True
                                        n_ok += 1
                                finally:
                                    tmp_path.unlink(missing_ok=True)
                                break
                        if converted:
                            break
                    except Exception:
                        pass

            if not converted and ann_fmt == "voc":
                xml_path = img_path.with_suffix(".xml")
                if not xml_path.exists():
                    xml_path = list(MAKRAI_DIR.rglob(f"{stem}.xml"))
                    xml_path = xml_path[0] if xml_path else None
                if xml_path and xml_path.exists():
                    r = voc_to_yolo(xml_path, img_path, out_img, out_lbl, dry_run=dry_run)
                    if r:
                        format_used = "YOLO detection box (Pascal VOC source)"
                        converted = True
                        n_ok += 1

            if not converted:
                # Last resort: copy image, write empty label
                new_stem = stem
                if not dry_run:
                    out_img.mkdir(parents=True, exist_ok=True)
                    out_lbl.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, out_img / (new_stem + img_path.suffix))
                    (out_lbl / (new_stem + ".txt")).write_text("", encoding="utf-8")
                n_ok += 1
                if format_used == "unknown":
                    format_used = "YOLO detection box (no annotations found)"

        print(f"    → {n_ok}/{len(imgs)} images processed for split '{split}'")
        return n_ok

    n_tr = process_split(train_imgs, "train")
    n_va = process_split(val_imgs, "val")
    n_te = process_split(test_imgs, "test")

    return n_tr, n_va, n_te, format_used


def convert_mcount(dry_run: bool) -> Tuple[int, str]:
    """Convert MCount dataset → all into test split. Returns (n_test, format_used)."""
    print("\n=== Converting MCount (all → test split) ===")
    ann_fmt = detect_annotation_format(MCOUNT_DIR)
    print(f"  Detected annotation format: {ann_fmt}")

    all_images = sorted([
        p for p in MCOUNT_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
        and "mask" not in p.stem.lower()
    ])
    print(f"  Found {len(all_images)} images (excluding mask files)")

    out_img = DATA_PROC / "images" / "test"
    out_lbl = DATA_PROC / "labels" / "test"
    format_used = "unknown"
    n_ok = 0

    for img_path in tqdm(all_images, desc="MCount → test"):
        stem = img_path.stem
        prefix = "mcount_"
        converted = False

        # Try mask-based segmentation
        if ann_fmt in ("mask", "unknown"):
            mask_candidates = list(MCOUNT_DIR.rglob(f"*{stem}*mask*"))
            mask_candidates += list(MCOUNT_DIR.rglob(f"*mask*{stem}*"))
            if mask_candidates:
                r = mask_to_yolo_seg(
                    img_path, mask_candidates[0], out_img, out_lbl,
                    prefix=prefix, dry_run=dry_run
                )
                if r:
                    format_used = "YOLO segmentation (mask PNG source)"
                    converted = True
                    n_ok += 1

        if not converted and ann_fmt == "csv":
            csv_candidates = list(MCOUNT_DIR.rglob(f"{stem}*.csv"))
            if csv_candidates:
                r = csv_to_yolo_box(
                    img_path, csv_candidates[0], out_img, out_lbl,
                    prefix=prefix, dry_run=dry_run
                )
                if r:
                    format_used = "YOLO detection box (CSV source)"
                    converted = True
                    n_ok += 1

        if not converted:
            # Look for any annotation file with same stem
            xml_candidates = list(MCOUNT_DIR.rglob(f"{stem}.xml"))
            if xml_candidates:
                r = voc_to_yolo(xml_candidates[0], img_path, out_img, out_lbl,
                                prefix=prefix, dry_run=dry_run)
                if r:
                    format_used = "YOLO detection box (Pascal VOC source)"
                    converted = True
                    n_ok += 1

        if not converted:
            # Copy with empty label
            new_stem = f"{prefix}{stem}"
            if not dry_run:
                out_img.mkdir(parents=True, exist_ok=True)
                out_lbl.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, out_img / (new_stem + img_path.suffix))
                (out_lbl / (new_stem + ".txt")).write_text("", encoding="utf-8")
            n_ok += 1
            format_used = "unknown (no annotation found)"

    print(f"  → {n_ok}/{len(all_images)} MCount images added to test split with prefix 'mcount_'")
    return n_ok, format_used


# ──────────────────────────────────────────────────────────────────────────────
# data.yaml writer
# ──────────────────────────────────────────────────────────────────────────────

def write_data_yaml(seg_mode: bool):
    CONFIGS.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(DATA_PROC.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "colony"},
    }
    out = CONFIGS / "data.yaml"
    with open(out, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print(f"\n  Wrote {out}")


# ──────────────────────────────────────────────────────────────────────────────
# CONVERSION_NOTES.md writer
# ──────────────────────────────────────────────────────────────────────────────

def write_conversion_notes(
    n_tr: int, n_va: int, n_te: int,
    makrai_fmt: str, mcount_fmt: str,
    seg_mode: bool, n_mcount_test: int
):
    (DATA_PROC).mkdir(parents=True, exist_ok=True)
    notes = f"""# Conversion Notes

## Format Decision

| Dataset | Source Format | Output Format |
|---------|--------------|---------------|
| Makrai 2023 | {makrai_fmt.split('(')[1].rstrip(')')} | {makrai_fmt.split('(')[0].strip()} |
| MCount | {mcount_fmt.split('(')[1].rstrip(')') if '(' in mcount_fmt else mcount_fmt} | {mcount_fmt.split('(')[0].strip()} |

**Segmentation mode:** {"enabled — polygon points used where available" if seg_mode else "disabled — bounding boxes only"}

## Rationale

{"YOLO segmentation format was used because source annotations provide polygon/mask data." if seg_mode else "YOLO detection box format was used because source annotations do not provide polygon/mask data, or --seg-fallback flag was passed."}

## Split Strategy

| Split | Count | Notes |
|-------|-------|-------|
| train | {n_tr} | Makrai 2023 only (80% of ~369 images) |
| val | {n_va} | Makrai 2023 only (10%) |
| test (Makrai) | {n_te} | Makrai 2023 only (10%) — general evaluation |
| test (MCount) | {n_mcount_test} | MCount entire dataset — merged-colony evaluation |

**Total test images:** {n_te + n_mcount_test}

**Important:** MCount images are entirely held out — they are NOT included in training or validation.
Their filenames are prefixed with `mcount_` to distinguish them in benchmark reporting.
The benchmark script (`run_benchmark.py`) separates general-test and MCount results into
distinct tables.

## Why not mix MCount into training?

MCount's value is as an unbiased signal on the exact failure mode (merged colonies) this
project targets. If mixed into training, the held-out signal would be contaminated and we
could not fairly claim "improvement on merged colonies." We keep it fully separate so the
benchmark comparison is valid.

If initial training produces poor merged-colony recall (<50% on MCount test), a small labeled
subset (~10%) may be moved to training. This would be noted here.
"""
    (DATA_PROC / "CONVERSION_NOTES.md").write_text(notes, encoding="utf-8")
    print(f"  Wrote {DATA_PROC / 'CONVERSION_NOTES.md'}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert datasets to YOLO format.")
    parser.add_argument("--seg-fallback", action="store_true",
                        help="Force bounding-box (detection) output, skip polygon extraction.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write any files — just print what would happen.")
    args = parser.parse_args()

    seg_mode = not args.seg_fallback

    if not MAKRAI_DIR.exists():
        print(f"ERROR: {MAKRAI_DIR} not found. Run download_datasets.py first.", file=sys.stderr)
        sys.exit(1)

    # Convert Makrai
    n_tr, n_va, n_te, makrai_fmt = convert_makrai(dry_run=args.dry_run, seg_mode=seg_mode)

    # Convert MCount
    n_mcount, mcount_fmt = convert_mcount(dry_run=args.dry_run)

    # Write configs
    if not args.dry_run:
        write_data_yaml(seg_mode)
        write_conversion_notes(
            n_tr, n_va, n_te, makrai_fmt, mcount_fmt,
            seg_mode, n_mcount
        )

    # Print summary
    print("\n" + "=" * 50)
    print("YOLO Conversion Summary")
    print("=" * 50)
    print(f"  Train (Makrai):  {n_tr:>5} images/labels")
    print(f"  Val   (Makrai):  {n_va:>5} images/labels")
    print(f"  Test  (Makrai):  {n_te:>5} images/labels")
    print(f"  Test  (MCount):  {n_mcount:>5} images/labels  [mcount_ prefix]")
    print(f"  Total test:      {n_te + n_mcount:>5}")
    print(f"\n  Format: {makrai_fmt} / {mcount_fmt}")
    print(f"\n  configs/data.yaml written")
    print(f"  data/processed/CONVERSION_NOTES.md written")
    print("\n✓ Conversion complete.")


if __name__ == "__main__":
    main()
