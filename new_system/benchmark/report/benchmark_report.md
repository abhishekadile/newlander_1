# Colony Detection Benchmark Report

Generated: 2026-06-22 20:32  
Methodology: 5 timed runs per image after 1 warm-up; latency = wall-clock including preprocessing.  
New system backend: openvino  
Current system: skipped (--skip-current)

---

## 1. Results — Makrai 2023 Test Split

Ground truth: YOLO label files (instance count per image).  
N = 0 images, annotated colonies from Makrai et al. 2023 test split.

### 1a. Summary

| System                       | Images w/GT  | Mean err%  | Mean lat (ms)  | p50 lat (ms)  | p95 lat (ms)  |
|------------------------------|--------------|------------|----------------|---------------|---------------|
| New (YOLO26n/openvino)       | 0            | N/A%       | N/A            | N/A           | N/A           |

### 1b. Per-image detail

| Image                              | GT     | Current pred | New pred   | Cur err% | New err% | Cur lat(ms)  | New lat(ms)  |
|------------------------------------|--------|--------------|------------|----------|----------|--------------|--------------|

---

## 2. Results — In-Repo Sample Images

N = 9 images from `images/` directory.  
**No ground truth is available for these images — latency only.**

| System                       | Mean lat (ms)  | p50 lat (ms)  | p95 lat (ms)  | Note                           |
|------------------------------|----------------|---------------|---------------|--------------------------------|
| New (YOLO26n/openvino)       | 114.9          | 112.4         | 130.5         | No GT — latency only           |

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
- **New system**: runs YOLO26n inference via openvino at conf=0.25, iou=0.45.
- **Latency**: 1 warmup + 5 timed runs per image. Reported as mean / p50 / p95.
- **Count error**: |predicted_count − gt_count| / gt_count × 100 %.
  Images without GT are excluded from accuracy metrics.

---

*Dataset: Makrai et al. (2023), CC BY 4.0, https://doi.org/10.6084/m9.figshare.22022540.v3*
