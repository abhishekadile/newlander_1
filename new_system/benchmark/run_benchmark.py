#!/usr/bin/env python3
"""
run_benchmark.py
================
Phase 6: Head-to-head benchmark — current OpenCFU system vs. new YOLO26 system.

Test sets:
  (a) General test set: Makrai 2023 test split images + repo sample images
  (b) MCount merged-colony subset: test images prefixed with "mcount_"

For each image:
  - 1 warm-up call (discarded)
  - 5 timed inference calls → mean, p50, p95 latency
  - Predicted count vs. ground-truth label → per-image error

Outputs:
  benchmark/report/benchmark_report.md  (comparison tables + summary)
  benchmark/report/benchmark_raw.json   (full per-image results)

Usage:
    cd new_system/
    python benchmark/run_benchmark.py [--skip-current] [--skip-new] [--n-runs 5]

Requirements:
    pip install requests ultralytics numpy tqdm
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print("ERROR: pip install numpy tqdm", file=sys.stderr)
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).resolve().parent
NEW_SYSTEM_ROOT = BENCHMARK_DIR.parent
REPO_ROOT = NEW_SYSTEM_ROOT.parent

DATA_PROC = NEW_SYSTEM_ROOT / "data" / "processed"
REPORT_DIR = BENCHMARK_DIR / "report"
REPORT_MD = REPORT_DIR / "benchmark_report.md"
REPORT_JSON = REPORT_DIR / "benchmark_raw.json"

REPO_IMAGES = REPO_ROOT / "images"


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth loading from YOLO labels
# ──────────────────────────────────────────────────────────────────────────────

def load_ground_truth(labels_dir: Path) -> Dict[str, int]:
    """
    Return {image_stem: colony_count} from YOLO label files.
    Empty label file = 0 colonies.
    """
    gt = {}
    if not labels_dir.exists():
        return gt
    for lbl_file in labels_dir.glob("*.txt"):
        lines = [l for l in lbl_file.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
        gt[lbl_file.stem] = len(lines)
    return gt


def collect_test_images(split_dir: Path, labels_dir: Path) -> List[Tuple[Path, int]]:
    """
    Collect (image_path, ground_truth_count) pairs from a split directory.
    """
    gt_map = load_ground_truth(labels_dir)
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    pairs = []
    if not split_dir.exists():
        return pairs
    for img in sorted(split_dir.iterdir()):
        if img.suffix.lower() in IMAGE_EXTS:
            gt_count = gt_map.get(img.stem, None)  # None = no ground truth
            pairs.append((img, gt_count))
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Timing runner
# ──────────────────────────────────────────────────────────────────────────────

def run_timed(adapter_fn, image_path: Path, n_runs: int = 5) -> dict:
    """
    Run adapter_fn(image_path) n_runs+1 times.
    Discard the first (warm-up), report stats on the remaining n_runs.
    """
    result = None
    latencies = []

    for i in range(n_runs + 1):
        try:
            t0 = time.perf_counter()
            result = adapter_fn(image_path)
            latency = (time.perf_counter() - t0) * 1000  # ms
            if i > 0:  # discard warm-up
                latencies.append(latency)
        except Exception as e:
            return {
                "error": str(e),
                "count": None,
                "latencies": [],
            }

    return {
        "count": result["count"] if result else None,
        "latencies": latencies,
        "colonies": result.get("colonies", []) if result else [],
        "error": None,
    }


def latency_stats(latencies: List[float]) -> dict:
    if not latencies:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": statistics.mean(latencies),
        "p50": statistics.median(latencies),
        "p95": float(np.percentile(latencies, 95)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-image error metric
# ──────────────────────────────────────────────────────────────────────────────

def count_error_pct(predicted: Optional[int], ground_truth: Optional[int]) -> Optional[float]:
    """Return percentage error relative to ground truth, or None if GT unavailable."""
    if predicted is None or ground_truth is None:
        return None
    if ground_truth == 0:
        return 0.0 if predicted == 0 else 100.0
    return abs(predicted - ground_truth) / ground_truth * 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(args):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Import adapters ────────────────────────────────────────────────────────
    sys.path.insert(0, str(BENCHMARK_DIR))
    current_fn = None
    new_fn = None

    if not args.skip_current:
        try:
            import current_system_adapter as cur
            current_fn = cur.run
            print("  ✓ Current system adapter loaded (OpenCFU/Node.js)")
        except Exception as e:
            print(f"  WARNING: current system adapter failed to load: {e}", file=sys.stderr)

    if not args.skip_new:
        try:
            import new_system_adapter as nw
            new_fn = nw.run
            print("  ✓ New system adapter loaded (YOLO26)")
        except Exception as e:
            print(f"  WARNING: new system adapter failed to load: {e}", file=sys.stderr)

    if current_fn is None and new_fn is None:
        print("ERROR: Both adapters failed to load.", file=sys.stderr)
        sys.exit(1)

    # ── Collect test images ────────────────────────────────────────────────────
    test_img_dir = DATA_PROC / "images" / "test"
    test_lbl_dir = DATA_PROC / "labels" / "test"

    all_test = collect_test_images(test_img_dir, test_lbl_dir)

    # Split into general (Makrai) vs MCount
    general_set = [(p, gt) for p, gt in all_test if not p.name.startswith("mcount_")]
    mcount_set = [(p, gt) for p, gt in all_test if p.name.startswith("mcount_")]

    # Add repo sample images (no ground truth labels from YOLO — use count=None)
    repo_images = []
    if REPO_IMAGES.exists():
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
        for img in sorted(REPO_IMAGES.iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                repo_images.append((img, None))
    general_set.extend(repo_images)

    print(f"\n  Test set sizes:")
    print(f"    General (Makrai test + repo samples): {len(general_set)}")
    print(f"    MCount (merged-colony):               {len(mcount_set)}")
    print(f"    n_runs per image:                      {args.n_runs} (+ 1 warm-up)")

    if not general_set and not mcount_set:
        print("  ERROR: No test images found. Run download_datasets.py and convert_to_yolo.py first.")
        sys.exit(1)

    # ── Run benchmark ──────────────────────────────────────────────────────────
    raw_results = {"general": [], "mcount": []}

    def run_set(img_set: List[Tuple[Path, Optional[int]]], label: str) -> List[dict]:
        print(f"\n=== Benchmarking: {label} ({len(img_set)} images) ===")
        records = []
        for img_path, gt_count in tqdm(img_set, desc=label):
            record = {
                "image": img_path.name,
                "ground_truth": gt_count,
                "current": None,
                "new": None,
            }

            if current_fn is not None:
                cur_result = run_timed(current_fn, img_path, n_runs=args.n_runs)
                record["current"] = {
                    "count": cur_result["count"],
                    "error_pct": count_error_pct(cur_result["count"], gt_count),
                    "latency": latency_stats(cur_result["latencies"]),
                    "error_msg": cur_result.get("error"),
                }

            if new_fn is not None:
                new_result = run_timed(new_fn, img_path, n_runs=args.n_runs)
                record["new"] = {
                    "count": new_result["count"],
                    "error_pct": count_error_pct(new_result["count"], gt_count),
                    "latency": latency_stats(new_result["latencies"]),
                    "error_msg": new_result.get("error"),
                }

            records.append(record)
        return records

    raw_results["general"] = run_set(general_set, "General")
    raw_results["mcount"] = run_set(mcount_set, "MCount merged-colony")

    # Save raw JSON
    REPORT_JSON.write_text(json.dumps(raw_results, indent=2), encoding="utf-8")
    print(f"\n  Raw results saved to {REPORT_JSON}")

    # ── Aggregate stats ────────────────────────────────────────────────────────
    def aggregate(records: List[dict]) -> dict:
        agg = {
            "n": len(records),
            "current": {"errors": [], "latency_means": [], "latency_p50s": [], "latency_p95s": []},
            "new":     {"errors": [], "latency_means": [], "latency_p50s": [], "latency_p95s": []},
        }
        for r in records:
            for system in ("current", "new"):
                sr = r.get(system)
                if sr is None:
                    continue
                if sr["error_pct"] is not None:
                    agg[system]["errors"].append(sr["error_pct"])
                lat = sr["latency"]
                if lat["mean"] is not None:
                    agg[system]["latency_means"].append(lat["mean"])
                    agg[system]["latency_p50s"].append(lat["p50"])
                    agg[system]["latency_p95s"].append(lat["p95"])
        return agg

    def safe_mean(lst):
        return statistics.mean(lst) if lst else None

    def fmt(val, unit=""):
        if val is None:
            return "N/A"
        return f"{val:.1f}{unit}"

    gen_agg = aggregate(raw_results["general"])
    mc_agg = aggregate(raw_results["mcount"])

    # ── Write Markdown report ──────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_runs_str = str(args.n_runs)

    def build_table(agg: dict, title: str) -> str:
        n = agg["n"]
        cur = agg["current"]
        new = agg["new"]

        cur_err = fmt(safe_mean(cur["errors"]), "%")
        new_err = fmt(safe_mean(new["errors"]), "%")
        cur_lat_mean = fmt(safe_mean(cur["latency_means"]), " ms")
        new_lat_mean = fmt(safe_mean(new["latency_means"]), " ms")
        cur_lat_p50 = fmt(safe_mean(cur["latency_p50s"]), " ms")
        new_lat_p50 = fmt(safe_mean(new["latency_p50s"]), " ms")
        cur_lat_p95 = fmt(safe_mean(cur["latency_p95s"]), " ms")
        new_lat_p95 = fmt(safe_mean(new["latency_p95s"]), " ms")

        return f"""
### {title} (n={n} images, {n_runs_str} timed runs + 1 warm-up)

| Metric | Current (OpenCFU) | New (YOLO26) | Δ |
|--------|------------------|-------------|---|
| Mean count error | {cur_err} | {new_err} | {"↓ improvement" if (safe_mean(new["errors"]) or 999) < (safe_mean(cur["errors"]) or 999) else "↑ regression" if safe_mean(cur["errors"]) is not None and safe_mean(new["errors"]) is not None else "—"} |
| Latency mean | {cur_lat_mean} | {new_lat_mean} | — |
| Latency p50 | {cur_lat_p50} | {new_lat_p50} | — |
| Latency p95 | {cur_lat_p95} | {new_lat_p95} | — |
"""

    # Plain-language summary
    gen_cur_err = safe_mean(gen_agg["current"]["errors"])
    gen_new_err = safe_mean(gen_agg["new"]["errors"])
    mc_cur_err = safe_mean(mc_agg["current"]["errors"])
    mc_new_err = safe_mean(mc_agg["new"]["errors"])
    gen_cur_lat = safe_mean(gen_agg["current"]["latency_means"])
    gen_new_lat = safe_mean(gen_agg["new"]["latency_means"])

    def _improvement(new_val, cur_val):
        if new_val is None or cur_val is None:
            return "data not available"
        delta = cur_val - new_val
        pct = delta / cur_val * 100 if cur_val else 0
        if delta > 0:
            return f"improved by {abs(pct):.1f}% ({cur_val:.1f} → {new_val:.1f})"
        else:
            return f"regressed by {abs(pct):.1f}% ({cur_val:.1f} → {new_val:.1f})"

    summary = (
        f"The new YOLO26-based system was benchmarked against the existing OpenCFU pipeline "
        f"on {gen_agg['n']} general test images and {mc_agg['n']} merged-colony images (MCount dataset). "
        f"On the general test set, counting accuracy {_improvement(gen_new_err, gen_cur_err)} "
        f"and latency {_improvement(gen_new_lat, gen_cur_lat)}. "
        f"On the merged-colony subset, accuracy {_improvement(mc_new_err, mc_cur_err)}. "
    )

    report = f"""# Colony Detection Benchmark Report

Generated: {ts}
Benchmark tool: `benchmark/run_benchmark.py`
n_runs per image: {n_runs_str} (+ 1 warm-up discarded)

---

## Summary

{summary}

> **Note:** "Mean count error" is the mean absolute percentage error (MAPE) relative to
> YOLO label ground truth. Lower is better. Images without ground-truth labels contribute
> to latency statistics only.

---

{build_table(gen_agg, "General Test Set (Makrai test split + repo sample images)")}

{build_table(mc_agg, "MCount Merged-Colony Subset (held-out evaluation)")}

---

## Per-Image Results (General)

| Image | GT | Current count | New count | Current err% | New err% | Current lat (ms) | New lat (ms) |
|-------|----|--------------|-----------|-------------|---------|-----------------|-------------|
"""
    for r in raw_results["general"]:
        cur = r.get("current") or {}
        nw = r.get("new") or {}
        report += (
            f"| {r['image']} | {r['ground_truth'] or '—'} "
            f"| {cur.get('count') or '—'} | {nw.get('count') or '—'} "
            f"| {fmt(cur.get('error_pct'))} | {fmt(nw.get('error_pct'))} "
            f"| {fmt((cur.get('latency') or {}).get('mean'))} "
            f"| {fmt((nw.get('latency') or {}).get('mean'))} |\n"
        )

    report += """
---

## Per-Image Results (MCount Merged-Colony)

| Image | GT | Current count | New count | Current err% | New err% | Current lat (ms) | New lat (ms) |
|-------|----|--------------|-----------|-------------|---------|-----------------|-------------|
"""
    for r in raw_results["mcount"]:
        cur = r.get("current") or {}
        nw = r.get("new") or {}
        report += (
            f"| {r['image']} | {r['ground_truth'] or '—'} "
            f"| {cur.get('count') or '—'} | {nw.get('count') or '—'} "
            f"| {fmt(cur.get('error_pct'))} | {fmt(nw.get('error_pct'))} "
            f"| {fmt((cur.get('latency') or {}).get('mean'))} "
            f"| {fmt((nw.get('latency') or {}).get('mean'))} |\n"
        )

    report += f"""
---

## Notes

- Latency is **wall-clock time** measured at the Python level (includes model loading overhead
  excluded via warm-up call). For production deployment, subtract warm-up amortized cost.
- The current system invokes OpenCFU via Node.js subprocess + Python preprocessing.
  Latency includes the full pipeline from raw image to parsed colony list.
- The new system uses the exported **OpenVINO** model for CPU-optimized inference.
  If OpenVINO model is missing, falls back to ONNX or PT.
- MCount results are reported **separately** from general results. Do not blend them —
  MCount specifically targets merged-colony scenarios, which inflate error rates
  for both systems relative to standard plates.

Raw data: `benchmark/report/benchmark_raw.json`
"""

    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"\n  ✓ Benchmark report written to {REPORT_MD}")
    print(f"\n  Quick summary:")
    print(f"    General — current err: {fmt(gen_cur_err, '%')}  new err: {fmt(gen_new_err, '%')}")
    print(f"    MCount  — current err: {fmt(mc_cur_err, '%')}  new err: {fmt(mc_new_err, '%')}")
    print(f"    General latency — current: {fmt(gen_cur_lat, 'ms')}  new: {fmt(gen_new_lat, 'ms')}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Head-to-head benchmark: OpenCFU vs. YOLO26.")
    parser.add_argument("--skip-current", action="store_true",
                        help="Skip current system (only benchmark new system).")
    parser.add_argument("--skip-new", action="store_true",
                        help="Skip new system (only benchmark current system).")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of timed inference calls per image (+ 1 warm-up). Default: 5.")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
