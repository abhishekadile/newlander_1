"""
run_benchmark.py

Head-to-head benchmark: current OpenCFU system vs. YOLO26n system.

Test images:
  1. Makrai test split (ground truth from YOLO label files)
  2. In-repo sample images (images/ directory — no ground truth available)

Per image: 1 warmup run + 5 timed runs → mean / p50 / p95 latency.
Metrics: predicted count vs. GT count → absolute error, relative error %.

Usage:
    python new_system/benchmark/run_benchmark.py \\
        [--test-dir new_system/data/processed/test] \\
        [--sample-dir images] \\
        [--output new_system/benchmark/report/benchmark_report.md] \\
        [--runs 5] \\
        [--skip-current]   # skip current system (if Express server unavailable)
"""

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Import adapters
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCHMARK_DIR))

import current_system_adapter as current_adapter
import new_system_adapter as new_adapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)


def pct(a: float, b: float) -> float:
    """Return a / b * 100, or nan if b == 0."""
    return (a / b * 100) if b != 0 else float("nan")


def percentile(data: list, p: float) -> float:
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def load_gt_count(label_path: Path) -> int:
    """Count lines in a YOLO .txt label file = number of annotated instances."""
    if not label_path.exists():
        return -1
    lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
    return len(lines)


def format_float(v: float, decimals: int = 1) -> str:
    if math.isnan(v) or math.isinf(v):
        return "N/A"
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# Per-image timing: 1 warmup + N timed runs
# ---------------------------------------------------------------------------

def time_system(run_fn, image_path: str, n_runs: int) -> tuple[list, dict]:
    """
    Returns (latencies_ms_list, result_from_last_run).
    The first call (warmup) is discarded from latencies.
    """
    # Warmup
    _ = run_fn(image_path)

    latencies = []
    last_result = {}
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = run_fn(image_path)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        last_result = result

    # Override latency with reported value from the last run (includes preprocessing)
    # but use our wall-clock latencies for p50/p95 consistency
    return latencies, last_result


# ---------------------------------------------------------------------------
# Benchmark one image
# ---------------------------------------------------------------------------

def benchmark_image(
    image_path: Path,
    gt_count: int,
    n_runs: int,
    skip_current: bool,
) -> dict:
    img_str = str(image_path)
    result = {
        "image":    image_path.name,
        "gt_count": gt_count,
        "current":  None,
        "new":      None,
    }

    # --- Current system ---
    if not skip_current:
        try:
            latencies, last = time_system(current_adapter.run, img_str, n_runs)
            pred = last.get("count", -1)
            err = abs(pred - gt_count) if gt_count >= 0 else float("nan")
            rel_err = pct(err, gt_count) if gt_count > 0 else float("nan")
            result["current"] = {
                "pred_count":  pred,
                "abs_error":   err,
                "rel_error_%": rel_err,
                "lat_mean_ms": statistics.mean(latencies),
                "lat_p50_ms":  percentile(latencies, 50),
                "lat_p95_ms":  percentile(latencies, 95),
                "latencies":   latencies,
                "error_msg":   last.get("error"),
                "mode":        last.get("mode", "?"),
            }
        except Exception as exc:
            result["current"] = {"error_msg": str(exc)}

    # --- New system ---
    try:
        new_adapter.warmup(img_str)
        latencies, last = time_system(new_adapter.run, img_str, n_runs)
        pred = last.get("count", -1)
        err = abs(pred - gt_count) if gt_count >= 0 else float("nan")
        rel_err = pct(err, gt_count) if gt_count > 0 else float("nan")
        result["new"] = {
            "pred_count":  pred,
            "abs_error":   err,
            "rel_error_%": rel_err,
            "lat_mean_ms": statistics.mean(latencies),
            "lat_p50_ms":  percentile(latencies, 50),
            "lat_p95_ms":  percentile(latencies, 95),
            "latencies":   latencies,
            "error_msg":   last.get("error"),
            "backend":     last.get("backend", "?"),
        }
    except Exception as exc:
        result["new"] = {"error_msg": str(exc)}

    return result


# ---------------------------------------------------------------------------
# Collect test images
# ---------------------------------------------------------------------------

def collect_makrai_test(test_dir: Path) -> list[tuple[Path, int]]:
    """
    Returns list of (image_path, gt_count) for the Makrai test split.
    Ground truth is read from paired .txt label files.
    """
    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"
    if not images_dir.exists():
        log(f"Makrai test images dir not found: {images_dir}")
        return []
    pairs = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        label = labels_dir / (img.stem + ".txt")
        gt = load_gt_count(label)
        pairs.append((img, gt))
    log(f"Makrai test split: {len(pairs)} images (GT from YOLO labels)")
    return pairs


def collect_sample_images(sample_dir: Path) -> list[tuple[Path, int]]:
    """
    Returns (image_path, -1) for all images in the sample directory.
    GT = -1 means ground truth is not available.
    """
    if not sample_dir.exists():
        log(f"Sample image directory not found: {sample_dir}")
        return []
    pairs = []
    for img in sorted(sample_dir.iterdir()):
        if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            pairs.append((img, -1))
    log(f"In-repo sample images: {len(pairs)} (no GT available)")
    return pairs


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def aggregate(results: list[dict], system: str) -> dict:
    """Compute aggregate latency and error stats for one system across all results."""
    lat_means, lat_p50s, lat_p95s = [], [], []
    abs_errs, rel_errs = [], []

    for r in results:
        s = r.get(system)
        if not s or s.get("error_msg"):
            continue
        if not math.isnan(s.get("lat_mean_ms", float("nan"))):
            lat_means.append(s["lat_mean_ms"])
            lat_p50s.append(s["lat_p50_ms"])
            lat_p95s.append(s["lat_p95_ms"])
        if r["gt_count"] >= 0 and not math.isnan(s.get("rel_error_%", float("nan"))):
            abs_errs.append(s["abs_error"])
            rel_errs.append(s["rel_error_%"])

    return {
        "n_images_timed":    len(lat_means),
        "n_images_with_gt":  len(rel_errs),
        "lat_mean_ms":       statistics.mean(lat_means)  if lat_means  else float("nan"),
        "lat_p50_ms":        statistics.mean(lat_p50s)   if lat_p50s   else float("nan"),
        "lat_p95_ms":        statistics.mean(lat_p95s)   if lat_p95s   else float("nan"),
        "mean_abs_error":    statistics.mean(abs_errs)   if abs_errs   else float("nan"),
        "mean_rel_error_%":  statistics.mean(rel_errs)   if rel_errs   else float("nan"),
    }


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _row(cols: list, widths: list) -> str:
    return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |"


def _sep(widths: list) -> str:
    return "|" + "|".join("-" * (w + 2) for w in widths) + "|"


def write_report(
    makrai_results: list[dict],
    sample_results: list[dict],
    output_path: Path,
    n_runs: int,
    skip_current: bool,
    new_backend: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    makrai_agg_cur = aggregate(makrai_results, "current")
    makrai_agg_new = aggregate(makrai_results, "new")
    sample_agg_cur = aggregate(sample_results, "current")
    sample_agg_new = aggregate(sample_results, "new")

    systems = []
    if not skip_current:
        systems.append(("Current (OpenCFU)", "current", makrai_agg_cur, sample_agg_cur))
    systems.append((f"New (YOLO26n/{new_backend})", "new", makrai_agg_new, sample_agg_new))

    # ------------------------------------------------------------------
    def table_header(cols, widths):
        return "\n".join([
            _row(cols, widths),
            _sep(widths),
        ])

    def per_image_table(results: list[dict], show_gt: bool) -> str:
        cols = ["Image", "GT", "Current pred", "New pred", "Cur err%", "New err%",
                "Cur lat(ms)", "New lat(ms)"]
        widths = [34, 6, 12, 10, 8, 8, 12, 12]
        lines = [_row(cols, widths), _sep(widths)]
        for r in results:
            gt = str(r["gt_count"]) if r["gt_count"] >= 0 else "—"
            cur = r.get("current") or {}
            nw  = r.get("new")     or {}

            cur_pred = str(cur.get("pred_count", "err")) if not cur.get("error_msg") else "error"
            new_pred = str(nw.get("pred_count",  "err")) if not nw.get("error_msg")  else "error"
            cur_err  = format_float(cur.get("rel_error_%", float("nan"))) + "%" if not cur.get("error_msg") else "—"
            new_err  = format_float(nw.get("rel_error_%",  float("nan"))) + "%" if not nw.get("error_msg")  else "—"
            cur_lat  = format_float(cur.get("lat_mean_ms", float("nan"))) if not cur.get("error_msg") else "—"
            new_lat  = format_float(nw.get("lat_mean_ms",  float("nan"))) if not nw.get("error_msg")  else "—"

            lines.append(_row(
                [r["image"][:34], gt, cur_pred, new_pred, cur_err, new_err, cur_lat, new_lat],
                widths,
            ))
        return "\n".join(lines)

    def summary_table(results_list: list) -> str:
        cols = ["System", "Images w/GT", "Mean err%", "Mean lat (ms)", "p50 lat (ms)", "p95 lat (ms)"]
        widths = [28, 12, 10, 14, 13, 13]
        lines = [_row(cols, widths), _sep(widths)]
        for name, _, m_agg, _ in systems:
            lines.append(_row([
                name,
                str(m_agg["n_images_with_gt"]),
                format_float(m_agg["mean_rel_error_%"]) + "%",
                format_float(m_agg["lat_mean_ms"]),
                format_float(m_agg["lat_p50_ms"]),
                format_float(m_agg["lat_p95_ms"]),
            ], widths))
        return "\n".join(lines)

    def sample_latency_table() -> str:
        cols = ["System", "Mean lat (ms)", "p50 lat (ms)", "p95 lat (ms)", "Note"]
        widths = [28, 14, 13, 13, 30]
        lines = [_row(cols, widths), _sep(widths)]
        for name, _, _, s_agg in systems:
            lines.append(_row([
                name,
                format_float(s_agg["lat_mean_ms"]),
                format_float(s_agg["lat_p50_ms"]),
                format_float(s_agg["lat_p95_ms"]),
                "No GT — latency only",
            ], widths))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    report = f"""# Colony Detection Benchmark Report

Generated: {run_date}  
Methodology: {n_runs} timed runs per image after 1 warm-up; latency = wall-clock including preprocessing.  
New system backend: {new_backend}  
Current system: {"skipped (--skip-current)" if skip_current else "OpenCFU via Express /detect or direct Node.js"}

---

## 1. Results — Makrai 2023 Test Split

Ground truth: YOLO label files (instance count per image).  
N = {len(makrai_results)} images, annotated colonies from Makrai et al. 2023 test split.

### 1a. Summary

{summary_table(makrai_results)}

### 1b. Per-image detail

{per_image_table(makrai_results, show_gt=True)}

---

## 2. Results — In-Repo Sample Images

N = {len(sample_results)} images from `images/` directory.  
**No ground truth is available for these images — latency only.**

{sample_latency_table()}

---

## 3. Named Limitations — Read Before Interpreting Results

These are not footnotes. They directly bear on whether "better than OpenCFU" claims
are fully substantiated.

### 3.1 Merged / touching colonies — NOT EVALUATED

The MCount dataset (Dryad), which contains images specifically designed to test
merged and touching colony scenarios, is **currently inaccessible** (locked as of
June 2026). This benchmark **cannot** evaluate either system's performance on
merged/touching colonies.

**Impact:** OpenCFU uses morphological analysis designed for connected blobs;
YOLO26n is trained only on single-colony bounding boxes. Neither system's
handling of merged colonies is validated here.

**Action required:** Revisit once MCount access is restored on Dryad, or once
sufficient touching/merged-colony images are collected from IncuCountAPI
production deployments.

### 3.2 Glare and variable lighting — NOT FULLY EVALUATED

Makrai et al. 2023 provides plate images with two background conditions: white
agar and black agar. Real-world illumination variation (overhead glare, outdoor
light, shadows) **beyond this binary variation is not represented** in training or
test data.

The `hsv_v=0.6` augmentation applied during training provides a partial proxy
for brightness variation but is not a substitute for genuine lighting-diversity
data.

### 3.3 Species and scene diversity

The training data covers 24 bacterial species across ~369 scene images.
Generalisation to species or plate types not present in Makrai 2023 is untested.

---

## 4. What Was Confirmed

- Detection accuracy (count error) on single, non-overlapping colonies from
  Makrai 2023 test split (white and black backgrounds, 24 species).
- Inference latency for both systems on the same images.
- Latency on 8 in-repo sample images (WIN_20250905, complex/standard count films, BMP).

## 5. What Remains Unverified

| Gap | Status |
|-----|--------|
| Merged / touching colony accuracy | NOT EVALUATED (MCount inaccessible) |
| Glare / variable lighting | NOT EVALUATED (no diverse lighting data) |
| Species outside Makrai 2023 | NOT EVALUATED |
| Segmentation-level accuracy | NOT APPLICABLE (bbox-only annotations) |

---

## 6. Methodology Notes

- **Current system**: calls `server.js`'s `/detect` route (HTTP) or falls back
  to a direct `ColonyDetector` Node.js subprocess. No modifications made.
- **New system**: runs YOLO26n inference via {new_backend} at conf=0.25, iou=0.45.
- **Latency**: 1 warmup + {n_runs} timed runs per image. Reported as mean / p50 / p95.
- **Count error**: |predicted_count − gt_count| / gt_count × 100 %.
  Images without GT are excluded from accuracy metrics.

---

*Dataset: Makrai et al. (2023), CC BY 4.0, https://doi.org/10.6084/m9.figshare.22022540.v3*
"""

    output_path.write_text(report, encoding="utf-8")
    log(f"Report written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head benchmark: OpenCFU vs. YOLO26n colony detection."
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed" / "test",
        help="Directory containing Makrai test split (expects images/ and labels/ subdirs).",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "images",
        help="Directory containing in-repo sample images (no GT).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "report" / "benchmark_report.md",
        help="Output path for the Markdown report.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timed runs per image (default: 5).",
    )
    parser.add_argument(
        "--skip-current",
        action="store_true",
        help="Skip current system (use when Express server is unavailable).",
    )
    args = parser.parse_args()

    log("=== Colony Detection Benchmark ===")
    log(f"Test dir  : {args.test_dir}")
    log(f"Sample dir: {args.sample_dir}")
    log(f"Output    : {args.output}")
    log(f"Runs/image: {args.runs} (+ 1 warmup)")
    log(f"Skip current: {args.skip_current}")
    log("")

    # Collect images
    makrai_pairs = collect_makrai_test(args.test_dir)
    sample_pairs = collect_sample_images(args.sample_dir)

    if not makrai_pairs and not sample_pairs:
        log("No test images found. Exiting.")
        sys.exit(1)

    # Determine new system backend for report header
    try:
        _, backend = new_adapter._find_model()
    except FileNotFoundError:
        backend = "not available"
        log(f"WARNING: New system model not found — new system results will show errors.")

    # Run benchmark
    makrai_results = []
    total = len(makrai_pairs)
    for i, (img, gt) in enumerate(makrai_pairs, 1):
        log(f"  Makrai [{i}/{total}] {img.name} (GT={gt}) …")
        r = benchmark_image(img, gt, args.runs, args.skip_current)
        makrai_results.append(r)
        cur_pred = (r.get("current") or {}).get("pred_count", "err")
        new_pred = (r.get("new") or {}).get("pred_count", "err")
        log(f"    cur={cur_pred}  new={new_pred}  gt={gt}")

    sample_results = []
    total = len(sample_pairs)
    for i, (img, gt) in enumerate(sample_pairs, 1):
        log(f"  Sample [{i}/{total}] {img.name} …")
        r = benchmark_image(img, gt, args.runs, args.skip_current)
        sample_results.append(r)

    # Save raw results JSON
    raw_out = args.output.parent / "benchmark_raw.json"
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    with raw_out.open("w") as fh:
        json.dump(
            {"makrai": makrai_results, "sample": sample_results},
            fh, indent=2, default=str,
        )
    log(f"Raw results JSON: {raw_out}")

    # Write Markdown report
    write_report(
        makrai_results, sample_results,
        args.output, args.runs, args.skip_current, backend,
    )

    log("")
    log("Benchmark complete.")


if __name__ == "__main__":
    main()
