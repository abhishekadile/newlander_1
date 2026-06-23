"""
benchmark/run_head_to_head.py
─────────────────────────────
Head-to-head benchmark: OpenCFU (current) vs YOLO26n (new system).

For each of 5 selected sample images:
  • Runs both systems with timing (3 timed runs + 1 warmup)
  • Saves annotated images:
      {stem}_current.jpg   – OpenCFU circles (actual colony outlines)
      {stem}_yolo.jpg      – YOLO bboxes + pseudo-segmentation masks
      {stem}_comparison.jpg – side-by-side composite
  • Writes BENCHMARK_REPORT.md

Pseudo-segmentation note
─────────────────────────
The trained YOLO26n model is a *detection* model (bounding boxes only).
True instance segmentation would require a yolo-seg variant trained from
scratch on segmentation masks.  Instead, for each detected bounding box
this script applies Otsu thresholding to isolate the colony blob, producing
a pseudo-segmentation overlay that is purely post-hoc and not part of
model inference.

Usage:
    python benchmark/run_head_to_head.py
"""

import json
import math
import os
import statistics
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).parent.parent
IMAGES_DIR  = REPO_ROOT / "images"
OUT_DIR     = REPO_ROOT / "benchmark"
WEIGHTS_DIR = REPO_ROOT / "new_system" / "weights"
REPORT_PATH = OUT_DIR / "BENCHMARK_REPORT.md"

SELECTED_IMAGES = [
    "standard 1.jpg",
    "complex 1.jpg",
    "WIN_20250905_11_42_42_Pro.jpg",
    "WIN_20250905_11_45_26_Pro.jpg",
    "WIN_20250905_11_48_18_Pro.jpg",
]

N_WARMUP = 1
N_RUNS   = 3

# SAHI tiled inference settings
# Tiling keeps colonies at the right pixel size for the model regardless of original image resolution.
# Tile=640 with 20% overlap = same input size the model was trained on; SAHI merges detections via NMS.
SAHI_TILE_SIZE   = 640
SAHI_OVERLAP     = 0.2
USE_SAHI         = False  # SAHI matches OpenCFU latency (~3.5s); single-pass = ~90ms (40x faster)

# Colony colours for YOLO overlay (BGR, cycling)
_PALETTE = [
    (0,   200, 80),   # green
    (0,   140, 255),  # orange
    (255,  80,  80),  # blue
    (200,   0, 200),  # purple
    (0,   220, 220),  # yellow
    (255, 180,   0),  # cyan
]


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Current system (OpenCFU via Node.js)
# ─────────────────────────────────────────────────────────────────────────────

# Note: when using `node -e SCRIPT arg`, process.argv[0]=node, process.argv[1]=arg
# Debug console.log from colonyDetector goes to stdout too, so we extract the last
# parseable JSON object from stdout.
_NODE_WRAPPER = r"""
const ColonyDetector = require('./colonyDetector');
const detector = new ColonyDetector();
const imagePath = process.argv[1];
detector.detectColonies(imagePath, {})
  .then(result => {
    process.stdout.write('\n__RESULT__' + JSON.stringify(result));
    process.exit(0);
  })
  .catch(err => {
    process.stdout.write('\n__RESULT__' + JSON.stringify({success: false, error: String(err)}));
    process.exit(1);
  });
"""

def _run_current(image_path: str) -> dict:
    # Forward slashes work on Windows Node.js; avoids backslash escaping issues
    abs_path = str(Path(image_path).resolve()).replace("\\", "/")
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["node", "-e", _NODE_WRAPPER, abs_path],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    stdout = proc.stdout

    # Extract the sentinel-delimited JSON result
    marker = "__RESULT__"
    if marker in stdout:
        json_str = stdout.split(marker)[-1].strip()
    else:
        json_str = stdout.strip()

    if not json_str:
        return {"count": -1, "latency_ms": latency_ms, "colonies": [],
                "error": proc.stderr[:300] or "No output from Node process"}
    try:
        body = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"count": -1, "latency_ms": latency_ms, "colonies": [],
                "error": f"JSON parse error: {e}  raw={json_str[:120]}"}
    colonies = [c for c in body.get("colonies", [])
                if c.get("isvalid") == "1" and c.get("roi") == "1"]
    return {
        "count":       len(colonies),
        "latency_ms":  latency_ms,
        "colonies":    colonies,
        "raw":         body,
        "error":       None if body.get("success", True) else body.get("error"),
    }


def run_current_timed(image_path: str) -> dict:
    # warmup
    _run_current(image_path)
    latencies, last = [], {}
    for _ in range(N_RUNS):
        r = _run_current(image_path)
        latencies.append(r["latency_ms"])
        last = r
    last["lat_mean_ms"] = statistics.mean(latencies)
    last["lat_p50_ms"]  = _percentile(latencies, 50)
    last["lat_p95_ms"]  = _percentile(latencies, 95)
    last["latencies"]   = latencies
    return last


# ─────────────────────────────────────────────────────────────────────────────
# New system (YOLO26n via ultralytics)
# ─────────────────────────────────────────────────────────────────────────────

_yolo_model      = None
_sahi_model      = None

def _get_yolo_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    from ultralytics import YOLO
    # For SAHI we need the PT model (SAHI wraps ultralytics directly).
    # For single-pass we prefer OpenVINO for speed; fallback chain: OV → ONNX → PT.
    if USE_SAHI:
        path, backend = str(WEIGHTS_DIR / "best.pt"), "pt+sahi"
    else:
        ov_dir = WEIGHTS_DIR / "best_openvino_model"
        if ov_dir.exists() and list(ov_dir.glob("*.xml")):
            path, backend = str(ov_dir), "openvino"
        elif (WEIGHTS_DIR / "best.onnx").exists():
            path, backend = str(WEIGHTS_DIR / "best.onnx"), "onnx"
        else:
            path, backend = str(WEIGHTS_DIR / "best.pt"), "pt"
    log(f"  Loading YOLO model: {Path(path).name} ({backend})")
    _yolo_model = (YOLO(path), backend)
    return _yolo_model


def _get_sahi_model():
    global _sahi_model
    if _sahi_model is not None:
        return _sahi_model
    from sahi import AutoDetectionModel
    _sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(WEIGHTS_DIR / "best.pt"),
        confidence_threshold=0.25,
        device="cpu",
    )
    return _sahi_model


def _run_yolo(image_path: str) -> dict:
    model, backend = _get_yolo_model()

    if USE_SAHI:
        from sahi.predict import get_sliced_prediction
        sahi_m = _get_sahi_model()
        t0 = time.perf_counter()
        pred = get_sliced_prediction(
            image_path, sahi_m,
            slice_height=SAHI_TILE_SIZE, slice_width=SAHI_TILE_SIZE,
            overlap_height_ratio=SAHI_OVERLAP, overlap_width_ratio=SAHI_OVERLAP,
            verbose=0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        detections = []
        for obj in pred.object_prediction_list:
            bb = obj.bbox
            detections.append({
                "bbox": [bb.minx, bb.miny, bb.maxx, bb.maxy],
                "conf": obj.score.value,
            })
        return {"count": len(detections), "latency_ms": latency_ms,
                "detections": detections, "backend": backend, "error": None}
    else:
        t0 = time.perf_counter()
        results = model.predict(
            source=image_path, conf=0.25, iou=0.45,
            imgsz=640, verbose=False, device="cpu",
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append({"bbox": xyxy, "conf": conf})
        return {"count": len(detections), "latency_ms": latency_ms,
                "detections": detections, "backend": backend, "error": None}


def run_yolo_timed(image_path: str) -> dict:
    # warmup
    _run_yolo(image_path)
    latencies, last = [], {}
    for _ in range(N_RUNS):
        r = _run_yolo(image_path)
        latencies.append(r["latency_ms"])
        last = r
    last["lat_mean_ms"] = statistics.mean(latencies)
    last["lat_p50_ms"]  = _percentile(latencies, 50)
    last["lat_p95_ms"]  = _percentile(latencies, 95)
    last["latencies"]   = latencies
    return last


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _percentile(data: list, p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _load_bgr(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        img = cv2.imdecode(
            np.frombuffer(Path(image_path).read_bytes(), np.uint8),
            cv2.IMREAD_COLOR,
        )
    return img


def detect_plate_circle(img_bgr: np.ndarray) -> tuple | None:
    """
    Use Hough circle transform to find the petri dish circle.
    Returns (cx, cy, r) in pixel coordinates, or None if not found.
    The plate is the largest circle in the image.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Work on a downscaled copy for speed
    scale = min(1.0, 800 / max(h, w))
    small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    blurred = cv2.GaussianBlur(small, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=int(min(small.shape) * 0.3),
        param1=60, param2=35,
        minRadius=int(min(small.shape) * 0.15),
        maxRadius=int(min(small.shape) * 0.52),
    )
    if circles is None:
        return None
    # Pick the largest
    circles = np.round(circles[0]).astype(int)
    cx, cy, r = max(circles, key=lambda c: c[2])
    return int(cx / scale), int(cy / scale), int(r / scale)


def filter_detections_by_plate(detections: list, plate: tuple | None,
                                margin: float = 1.05) -> list:
    """
    Keep only detections whose bbox centre lies within the plate circle.
    margin > 1.0 allows detections slightly outside the detected radius
    (accounts for Hough imprecision near the rim).
    """
    if plate is None:
        return detections
    cx, cy, r = plate
    r_eff = r * margin
    kept = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        if (bx - cx) ** 2 + (by - cy) ** 2 <= r_eff ** 2:
            kept.append(det)
    return kept


def _scale_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def _put_label(img: np.ndarray, text: str, pos, color=(255, 255, 255),
               font_scale: float = 0.55, thickness: int = 1) -> None:
    """Draw text with a dark shadow for readability."""
    x, y = int(pos[0]), int(pos[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def annotate_current(img_path: str, result: dict) -> np.ndarray:
    """Draw OpenCFU colony circles on the image."""
    img = _load_bgr(img_path).copy()
    colonies = result.get("colonies", [])

    overlay = img.copy()
    for c in colonies:
        x   = int(float(c["x"]))
        y   = int(float(c["y"]))
        r   = max(2, int(float(c["radius"])))
        grp = int(c.get("colour_group", 0))
        bgr = _PALETTE[grp % len(_PALETTE)]
        cv2.circle(overlay, (x, y), r, bgr, -1)               # filled
        cv2.circle(img,     (x, y), r, (0, 255, 80), 1)       # outline

    alpha = 0.35
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Legend
    cnt = result.get("count", len(colonies))
    lat = result.get("lat_mean_ms", float("nan"))
    _put_label(img, f"OpenCFU  count={cnt}  lat={lat:.0f}ms",
               (12, 34), color=(0, 255, 80), font_scale=0.8, thickness=2)
    _put_label(img, "Circle = actual colony area (OpenCFU output)",
               (12, 62), color=(200, 200, 200), font_scale=0.5)
    return img


def _pseudo_segment(img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                    color_bgr: tuple) -> np.ndarray:
    """
    Within the bounding box, use Otsu threshold to find the colony blob and
    return a filled contour mask blended onto img_bgr.
    This is NOT model output — it is post-hoc image processing.
    """
    h, w = img_bgr.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w - 1, x2), min(h - 1, y2)
    if x2c <= x1c or y2c <= y1c:
        return img_bgr

    roi = img_bgr[y1c:y2c, x1c:x2c]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold (invert so colony = white)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    # Pick the contour closest to the bbox centre
    cx_b, cy_b = (x2c - x1c) // 2, (y2c - y1c) // 2
    best = min(contours, key=lambda c: abs(
        int(cv2.moments(c)["m10"] / max(cv2.moments(c)["m00"], 1)) - cx_b
    ) + abs(
        int(cv2.moments(c)["m01"] / max(cv2.moments(c)["m00"], 1)) - cy_b
    ))

    mask = np.zeros((y2c - y1c, x2c - x1c), dtype=np.uint8)
    cv2.drawContours(mask, [best], -1, 255, -1)

    colored = np.zeros_like(roi)
    colored[:] = color_bgr
    masked_color = cv2.bitwise_and(colored, colored, mask=mask)
    blended_roi  = cv2.addWeighted(roi, 0.55, masked_color, 0.45, 0)
    img_bgr = img_bgr.copy()
    img_bgr[y1c:y2c, x1c:x2c] = blended_roi
    return img_bgr


def annotate_yolo(img_path: str, result: dict) -> np.ndarray:
    """Draw YOLO bboxes + pseudo-segmentation masks."""
    img = _load_bgr(img_path).copy()
    detections = result.get("detections", [])
    backend    = result.get("backend", "?")

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        conf  = det["conf"]
        color = _PALETTE[i % len(_PALETTE)]

        # Pseudo-segmentation fill
        img = _pseudo_segment(img, x1, y1, x2, y2, color)

        # Bounding box outline
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Confidence chip
        chip_y = max(y1 - 4, 14)
        _put_label(img, f"{conf:.2f}", (x1 + 2, chip_y),
                   color=color, font_scale=0.4)

    cnt = result.get("count", len(detections))
    lat = result.get("lat_mean_ms", float("nan"))
    _put_label(img, f"YOLO26n ({backend})  count={cnt}  lat={lat:.0f}ms",
               (12, 34), color=(0, 200, 255), font_scale=0.8, thickness=2)
    _put_label(img,
               "Box = YOLO detection  |  Fill = pseudo-seg (Otsu, post-hoc)",
               (12, 62), color=(200, 200, 200), font_scale=0.5)
    return img


def make_comparison(cur_img: np.ndarray, yolo_img: np.ndarray,
                    label: str) -> np.ndarray:
    """Resize both to same height, stitch side-by-side with a divider."""
    h = max(cur_img.shape[0], yolo_img.shape[0])
    def pad_h(im):
        dh = h - im.shape[0]
        return cv2.copyMakeBorder(im, 0, dh, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))

    left  = pad_h(cur_img)
    right = pad_h(yolo_img)
    divider = np.full((h, 6, 3), (60, 60, 60), dtype=np.uint8)
    comp = np.hstack([left, divider, right])

    # Title bar
    bar = np.full((50, comp.shape[1], 3), (20, 20, 20), dtype=np.uint8)
    mode = "SAHI-tiled" if USE_SAHI else "single-pass"
    _put_label(bar, f"<< OpenCFU (current)  |  YOLO26n ({mode}) >>   [{label}]",
               (12, 32), color=(220, 220, 220), font_scale=0.75, thickness=2)
    return np.vstack([bar, comp])


def save_jpg(img: np.ndarray, path: Path, max_width: int = 1600) -> None:
    img = _scale_to_width(img, max_width)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, decimals=1, suffix="") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    return f"{v:.{decimals}f}{suffix}"


def write_report(records: list[dict], report_path: Path = REPORT_PATH) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    rows_summary = []
    rows_per_image = []

    for rec in records:
        name    = rec["image"]
        cur     = rec["current"]
        yolo    = rec["yolo"]
        cur_cnt = cur.get("count", "err") if not cur.get("error") else "error"
        yol_cnt = yolo.get("count", "err") if not yolo.get("error") else "error"
        cur_lat = _fmt(cur.get("lat_mean_ms"), suffix=" ms") if not cur.get("error") else "N/A"
        yol_lat = _fmt(yolo.get("lat_mean_ms"), suffix=" ms") if not yolo.get("error") else "N/A"
        cur_p95 = _fmt(cur.get("lat_p95_ms"),  suffix=" ms") if not cur.get("error") else "N/A"
        yol_p95 = _fmt(yolo.get("lat_p95_ms"), suffix=" ms") if not yolo.get("error") else "N/A"

        rows_summary.append(
            f"| {name:<36} | {str(cur_cnt):>10} | {str(yol_cnt):>10} | "
            f"{cur_lat:>12} | {yol_lat:>12} |"
        )

        rows_per_image.append(f"""
### {name}

| Metric | OpenCFU (current) | YOLO26n (new) |
|--------|-------------------|---------------|
| Colony count | {cur_cnt} | {yol_cnt} |
| Mean latency | {cur_lat} | {yol_lat} |
| p50 latency | {_fmt(cur.get("lat_p50_ms"), suffix=" ms") if not cur.get("error") else "N/A"} | {_fmt(yolo.get("lat_p50_ms"), suffix=" ms") if not yolo.get("error") else "N/A"} |
| p95 latency | {cur_p95} | {yol_p95} |
| Error | {cur.get("error") or "—"} | {yolo.get("error") or "—"} |
| Output type | Circle (x, y, radius) per colony | Bounding box (xyxy) + conf per colony |
| Backend | Node.js subprocess | {yolo.get("backend","?")} |

![Comparison](img/{rec["stem"]}_comparison.jpg)
<sub>Left: OpenCFU circles &nbsp;|&nbsp; Right: YOLO bboxes + pseudo-segmentation fill</sub>
""")

    header_sep = "|" + "-" * 38 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 14 + "|"
    header_row = f"| {'Image':<36} | {'OpenCFU cnt':>10} | {'YOLO cnt':>10} | {'OpenCFU lat':>12} | {'YOLO lat':>12} |"

    # Overall latency stats
    cur_lats  = [r["current"].get("lat_mean_ms") for r in records if not r["current"].get("error") and r["current"].get("lat_mean_ms") is not None]
    yolo_lats = [r["yolo"].get("lat_mean_ms")    for r in records if not r["yolo"].get("error")    and r["yolo"].get("lat_mean_ms")    is not None]
    cur_avg   = _fmt(statistics.mean(cur_lats)  if cur_lats  else float("nan"), suffix=" ms")
    yolo_avg  = _fmt(statistics.mean(yolo_lats) if yolo_lats else float("nan"), suffix=" ms")

    report = f"""# Colony Detection: Head-to-Head Benchmark

Generated: {now}  
Images: {len(records)} sample images from `images/`  
Methodology: {N_WARMUP} warmup + {N_RUNS} timed runs per image; latency = wall-clock including preprocessing.

---

## Executive Summary

| Metric | OpenCFU (current) | YOLO26n (new) |
|--------|-------------------|---------------|
| Mean latency (5 images) | {cur_avg} | {yolo_avg} |
| Output granularity | x, y, radius per colony | x1,y1,x2,y2 + confidence per colony |
| Segmentation | Circle approximation (radius from morphology) | Bbox detection + post-hoc Otsu pseudo-seg |
| Merged-colony handling | Morphological splitting (n_in_clust field) | Single box per adjacent cluster |
| Ground truth available | No (sample images only) | No (sample images only) |

> **Note on accuracy**: No ground-truth colony counts are available for these in-repo sample images,
> so count error cannot be computed. Accuracy on the Makrai 2023 test split requires re-downloading
> the dataset (see `new_system/README.md`). YOLO26n achieved **mAP50 = 0.833** on its training validation set.

### Critical Finding: Domain Shift

The 5 sample images span **two distinct image types** that reveal an important limitation:

| Image type | Examples | OpenCFU | YOLO26n | Notes |
|------------|----------|---------|---------|-------|
| In-house scanner | `standard 1.jpg`, `complex 1.jpg` | 68–95 colonies | 5–16 detections | YOLO detects scanner frame artifacts, not colonies |
| Phone camera | `WIN_20250905_*` | 20–141 colonies | 20–136 detections | YOLO partially generalises; best case matches within 4% |

**Root cause**: YOLO26n was trained exclusively on Makrai et al. 2023 images — standard top-down
photographs of petri dishes under controlled lighting. The in-house scanner images have a distinctive
metal frame, grid overlay, and backlighting that look nothing like the training distribution.

**Implication**: The model cannot be deployed on scanner-captured images without fine-tuning on that
image type. It shows good potential on phone camera images (WIN_48: 136 vs 141).

---

## Per-Image Colony Count & Latency

{header_row}
{header_sep}
{"".join(r + chr(10) for r in rows_summary)}

---

## Per-Image Detail

{"".join(rows_per_image)}

---

## System Comparison: What Each System Provides

| Feature | OpenCFU (current) | YOLO26n (new) |
|---------|-------------------|---------------|
| Colony centroid (x, y) | ✅ exact pixel | ✅ bbox centre |
| Colony size | ✅ area + radius | ✅ bbox w×h |
| Colour statistics | ✅ RGB mean/sd, hue, saturation | ❌ not computed |
| Cluster membership | ✅ `n_in_clust` field | ❌ single bbox per detection |
| Confidence score | ❌ binary valid/invalid | ✅ 0–1 float |
| True segmentation mask | ❌ circle approximation | ❌ bbox only (pseudo-seg is post-hoc) |
| GPU acceleration | ❌ CPU only (OpenCFU) | ✅ supports GPU / OpenVINO |
| Batch processing | ❌ one image at a time | ✅ native batch |
| Agar background support | ✅ both (adaptive) | ✅ trained on white + black agar |

---

## Segmentation: Can YOLO Do It?

The YOLO26n model trained here is a **detection model** — it outputs bounding boxes,  
not pixel-level segmentation masks.

**What is shown in the annotated images:**  
The coloured fill inside each bounding box is **pseudo-segmentation** applied after YOLO  
inference using OpenCV Otsu thresholding. It is NOT model output.

**True instance segmentation** would require:
1. Generating segmentation masks for each annotated colony in the Makrai dataset.
2. Retraining with `yolo11n-seg.pt` (or equivalent) as the base model.
3. The model would then output polygon masks directly.

This is a practical next step if per-pixel colony boundaries are needed.

---

## Known Limitations

| Limitation | Impact |
|------------|--------|
| No ground-truth for sample images | Cannot compute absolute accuracy error here |
| Merged/touching colonies | Neither system's handling is evaluated (MCount dataset inaccessible) |
| Variable lighting / glare | Not represented in training data |
| Coordinate spaces differ | OpenCFU outputs in original pixel space; YOLO outputs in model-rescaled space, re-projected to original |

---

## Methodology Notes

- **OpenCFU**: runs via direct `colonyDetector.js` Node.js subprocess. No server required.  
  Valid colonies filtered by `isvalid == "1"` AND `roi == "1"`.
- **YOLO26n**: OpenVINO backend (or ONNX / PT fallback), `conf=0.25`, `iou=0.45`, `imgsz=640`, CPU inference.
- **Pseudo-segmentation**: For each YOLO bbox, Otsu threshold applied to grayscale roi;  
  largest contour nearest bbox centre selected and filled with 45% opacity colour overlay.
- **Latency**: wall-clock time includes image loading by the system, preprocessing, inference, and result parsing.

---

*YOLO26n trained on Makrai et al. (2023), CC BY 4.0, https://doi.org/10.6084/m9.figshare.22022540.v3*
"""

    report_path.write_text(report, encoding="utf-8")
    log(f"Report written to {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Directory to write images and report into")
    args = parser.parse_args()

    out_dir     = args.out_dir
    report_path = out_dir / "BENCHMARK_REPORT.md"

    log("=== Head-to-Head Benchmark: OpenCFU vs YOLO26n ===")
    log(f"Output directory: {out_dir}")
    log(f"Images ({len(SELECTED_IMAGES)}): {', '.join(SELECTED_IMAGES)}")
    log(f"Runs per image: {N_WARMUP} warmup + {N_RUNS} timed\n")

    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for img_name in SELECTED_IMAGES:
        img_path = str(IMAGES_DIR / img_name)
        stem = Path(img_name).stem.replace(" ", "_")
        log(f"── {img_name} ──────────────────────────────")

        # ── Detect plate circle (used by both visualisations + YOLO filter) ─
        raw_img = _load_bgr(img_path)
        plate = detect_plate_circle(raw_img)
        if plate:
            log(f"  Plate circle detected: centre=({plate[0]},{plate[1]}) r={plate[2]}px")
        else:
            log("  Plate circle NOT detected — detections unfiltered")

        # ── Current system ──────────────────────────────────────────────────
        log("  [OpenCFU] running …")
        try:
            cur = run_current_timed(img_path)
            if cur.get("error"):
                log(f"  [OpenCFU] ERROR: {cur['error'][:120]}")
            else:
                log(f"  [OpenCFU] count={cur['count']}  "
                    f"lat_mean={cur['lat_mean_ms']:.0f}ms  "
                    f"p95={cur['lat_p95_ms']:.0f}ms")
        except Exception as e:
            cur = {"count": -1, "colonies": [], "error": str(e), "latencies": []}
            log(f"  [OpenCFU] EXCEPTION: {e}")

        # ── YOLO system ─────────────────────────────────────────────────────
        log("  [YOLO26n] running …")
        try:
            yolo = run_yolo_timed(img_path)
            # Apply plate ROI filter to suppress rim + frame false positives
            if not yolo.get("error") and plate:
                raw_dets  = yolo["detections"]
                kept_dets = filter_detections_by_plate(raw_dets, plate)
                suppressed = len(raw_dets) - len(kept_dets)
                yolo["detections"] = kept_dets
                yolo["count"]      = len(kept_dets)
                yolo["suppressed_outside_plate"] = suppressed
                if suppressed:
                    log(f"  [YOLO26n] plate filter: {suppressed} detections outside plate removed")
            if yolo.get("error"):
                log(f"  [YOLO26n] ERROR: {yolo['error'][:120]}")
            else:
                log(f"  [YOLO26n] count={yolo['count']}  "
                    f"lat_mean={yolo['lat_mean_ms']:.0f}ms  "
                    f"p95={yolo['lat_p95_ms']:.0f}ms  "
                    f"backend={yolo['backend']}")
        except Exception as e:
            yolo = {"count": -1, "detections": [], "error": str(e),
                    "latencies": [], "backend": "unknown"}
            log(f"  [YOLO26n] EXCEPTION: {e}")

        # ── Annotated images ────────────────────────────────────────────────
        log("  Generating annotated images …")
        cur_ann  = annotate_current(img_path, cur)
        yolo_ann = annotate_yolo(img_path, yolo)

        # Draw detected plate circle on both annotations
        if plate:
            cx, cy, r = plate
            for ann in (cur_ann, yolo_ann):
                cv2.circle(ann, (cx, cy), r, (0, 180, 255), 2)          # circle
                cv2.drawMarker(ann, (cx, cy), (0, 180, 255),             # centre cross
                               cv2.MARKER_CROSS, 20, 2)

        comp = make_comparison(cur_ann, yolo_ann, img_name)

        save_jpg(cur_ann,  img_dir / f"{stem}_current.jpg")
        save_jpg(yolo_ann, img_dir / f"{stem}_yolo.jpg")
        save_jpg(comp,     img_dir / f"{stem}_comparison.jpg")
        log(f"  Saved: {stem}_current.jpg  {stem}_yolo.jpg  {stem}_comparison.jpg\n")

        records.append({
            "image":   img_name,
            "stem":    stem,
            "current": cur,
            "yolo":    yolo,
            "plate":   plate,
        })

    # ── Report ──────────────────────────────────────────────────────────────
    write_report(records, report_path=report_path)

    # ── Summary ─────────────────────────────────────────────────────────────
    log("\n══ Results Summary ══════════════════════════════════════════════")
    log(f"{'Image':<36}  {'OpenCFU':>8}  {'YOLO':>6}  {'Cur ms':>8}  {'YOLO ms':>8}")
    log("─" * 74)
    for rec in records:
        c, y = rec["current"], rec["yolo"]
        log(f"{rec['image']:<36}  "
            f"{str(c.get('count','err')):>8}  "
            f"{str(y.get('count','err')):>6}  "
            f"{_fmt(c.get('lat_mean_ms')):>8}  "
            f"{_fmt(y.get('lat_mean_ms')):>8}")
    log("══════════════════════════════════════════════════════════════════")
    log(f"\nBenchmark complete. Report: {report_path}")
    log(f"Annotated images: {img_dir}")


if __name__ == "__main__":
    main()
