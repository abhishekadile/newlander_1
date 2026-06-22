# 04 — Accuracy

## Caveats

> **This codebase contains no automated test suite and no formal accuracy benchmarking.**  
> The accuracy analysis below is derived from code inspection, algorithm analysis, and the sample output CSV (`bacterial_colonies.csv`).

---

## 1. Ground Truth & Validation

### Available Ground Truth

| Source | Description |
|--------|-------------|
| `core_engine/bacterial_colonies.csv` | 129-colony sample output from a real plate (demo) |
| `core_engine/src/classifier/trainnedClassifier.xml` | Pre-trained Random Forest (2.6 MB) |
| Colony profiles | 4 validated plate type presets with lab-tuned parameters |

### OpenCFU Published Accuracy (Upstream)

OpenCFU was originally published with the following accuracy claims (from the original OpenCFU paper, Geissmann 2013):

- **Counting accuracy within ±5%** of manual count on standard bacterial colony plates
- **Pearson correlation R > 0.99** between automated and manual counts across a range of colony densities
- **Sensitivity** varies with colony density — performance degrades at very high confluency (>300 colonies/plate in crowded conditions)

> ⚠️ **These figures apply to the original OpenCFU library.** The IncuCountAPI applies additional preprocessing that may improve accuracy on non-standard images (e.g., phone camera photos).

---

## 2. Accuracy Factors

### 2.1 Classifier Quality (Random Forest)

The system uses **two trained Random Forest classifiers**:

| Classifier | Role | File |
|-----------|------|------|
| Pass 1 RF | Initial candidate filtering from multi-threshold contours | `trainnedClassifier.xml` (2.6 MB) |
| Pass 2 RF (post-split) | Final classification of single vs. multi-colony blobs | `trainnedClassifier_ps.xml` |

**RF configuration:**
```
Max depth:    10    (limits overfitting)
Min samples:  10    (prevents small leaf nodes)
Max trees:    100
Termination:  EPS=0.01 or MAX_ITER
Max categories: 3  (binary + noise class)
```

**Feature vector (13 features):**

| # | Feature | Meaning |
|---|---------|---------|
| 0 | `P²/A` | Compactness / circularity inverse |
| 1 | `(hull_area - area) / hull_area` | Convexity defect |
| 2 | `(hull_perim - perim) / hull_perim` | Perimeter convexity |
| 3 | `area_hole / total_area` | Hole area fraction |
| 4 | `perim_hole / total_perim` | Hole perimeter fraction |
| 5 | `W / (W + H)` | Aspect ratio |
| 6–12 | 7 Hu moments | Shape invariants (rotation/scale/translation) |

**Strengths:**
- Shape-based features are robust to colour variation and lighting changes
- Hu moments provide rotation invariance (important for non-symmetric colonies)
- Convexity/compactness features distinguish real circular colonies from noise/agar artefacts

**Weaknesses:**
- Classifier was trained on a **specific dataset** — performance on unseen plate types may degrade
- No retraining pathway exposed via the API
- The 13-feature set does not include colour features, so the classifier is blind to colony colour at the classification step (colour is only used in the downstream DBSCAN step)

---

### 2.2 Threshold Sensitivity

The `threshold_value` parameter (default: 15) is the single most impactful accuracy parameter.

| Threshold | Effect |
|-----------|--------|
| Too low (<10) | Over-detection — background noise classified as colonies |
| 15 (default) | Balanced for most standard colony plates |
| Too high (>25) | Under-detection — small/faint colonies missed |
| 5 (Nutrient Plates preset) | Tuned for lighter colonies on nutrient agar |

The illumination correction in Step 2 reduces threshold sensitivity, but the system is still sensitive to:
- **Very uneven lighting** — shadows across the plate
- **Reflective/glossy agar** — specular highlights cause false positives
- **High colony density** — overlapping colonies are harder to split

---

### 2.3 ROI Detection Accuracy

The Hough Circle Transform for dish detection can fail or be inaccurate when:

| Condition | Impact |
|-----------|--------|
| Image clutter (labels, ruler, hand) near dish edge | False circle detection |
| Very bright/dark background | Hough edge detection confused |
| Dish fills almost entire frame | `max_radius` constraint may reject correct circle |
| Tilted or non-circular dish | Hough circles cannot model ellipses |

The system addresses this with:
- **Three-tier fallback** (upscaled → original → legacy 256px)
- **Hardcoded circles** for known demo images
- **1.02× tolerance** when filtering colonies near the dish edge

---

### 2.4 Coordinate Mapping Accuracy

After preprocessing (upscale + crop), colony positions in the processed image are mapped back to original image coordinates. The mapping is:

```
x_orig = (x_proc + crop_offset_x) / scale_factor
```

**Potential error sources:**
- Rounding in scale factor computation (float64, minimal)
- Rounding in `int(round(...))` during circle detection ← small but non-zero
- If scale_factor is computed from an intermediate upscaled size rather than the true ratio

---

## 3. Sample Output Analysis

From `core_engine/bacterial_colonies.csv` (129 colonies):

```
IsValid,X,Y,ROI,Colour_group,N_in_clust,Area,Radius,Hue,Sat,Rmean,...
```

### Colony size distribution (from CSV):

| Metric | Value |
|--------|-------|
| Total colonies detected | 129 |
| Min radius | 4 px (original space) |
| Max radius | 21 px |
| Median radius | ~13 px |
| Min area | 23 px² |
| Max area | 1384 px² |

### Colour distribution:

All 129 colonies in the sample have `Colour_group = 0` (DBSCAN disabled) and `ROI = 1`.

Colony mean RGB clusters around `(130–150, 128–148, 120–140)` — grey/beige tones consistent with nutrient agar plates. A few outliers at `(55–90, 60–95, 65–105)` suggest colonies in the outer dish edge region where background bleed occurs.

### N_in_clust values:

| N_in_clust | Count (approx) |
|------------|---------------|
| 1 | ~95 (74%) |
| 2 | ~27 (21%) |
| 3 | ~7 (5%) |

~26% of detections involved blob splitting, which is normal for moderately dense plates.

---

## 4. Known Accuracy Limitations

| Limitation | Severity | Source |
|-----------|----------|--------|
| No automated test suite | 🔴 High | No accuracy metric can be computed reliably |
| Classifier trained on limited dataset | 🟡 Medium | Unknown generalisation to new plate types |
| Uploaded files not cleaned up | 🟡 Medium | Not accuracy-related but operational risk |
| Stale CSV possible if two concurrent requests race | 🟡 Medium | `colonyDetector.js` single instance, not thread-safe |
| No confidence score per colony | 🟡 Medium | RF output is categorical, not probabilistic |
| Hardcoded circles for demo images | 🟢 Low | Only affects 8 specific known images |
| 1-pixel rounding in coordinate mapping | 🟢 Low | Negligible at typical resolutions |

---

## 5. Recommended Accuracy Improvements

1. **Add a test suite** with ground-truth annotated images and known colony counts
2. **Expose RF confidence scores** — `cv::ml::RTrees::predict()` with `cv::ml::StatModel::RAW_OUTPUT` flag can return per-tree vote counts
3. **Add Otsu auto-threshold option** as the default for new plate types
4. **Implement file cleanup** to prevent disk exhaustion from uploaded test images
5. **Concurrency protection** — use a queue or mutex around the CSV write/read cycle since `bacterial_colonies.csv` is shared state
6. **Expose retraining endpoint** or at minimum a mechanism to replace the XML classifier files

---

## 6. Accuracy vs. Colony Count Profile Mapping

| Profile | Expected Accuracy Notes |
|---------|------------------------|
| Anaerobic Count Film | Standard small-colony detection. Default params → moderate accuracy |
| Coliform (CC) Film | Same as anaerobic; CC films have consistent colony morphology |
| MacConkey Plates | Color grouping ON improves differentiation of lactose-fermenting (pink) vs. non-fermenting (colourless) colonies |
| Nutrient Plates | Lower threshold (5) and larger max radius (185) needed for larger/lighter colonies on rich media |
