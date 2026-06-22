# 02 — Algorithms

## Overview

The detection pipeline uses a **two-phase machine learning approach** built on OpenCV, combined with classical image processing algorithms. The C++ core (OpenCFU) is supplemented by a Python preprocessing layer that implements additional algorithms.

---

## 1. Preprocessing Algorithms (`preprocess.py`)

### 1.1 Image Upscaling — Bicubic + Unsharp Mask

**Purpose:** Normalise all input images to a minimum width of 6000 px, matching the internal assumptions of OpenCFU's C++ thresholding pipeline.

**Algorithm:**
1. Compute scale factor: `scale = 6000 / original_width` (only if `width < 6000`)
2. Resize with **bicubic interpolation** (`cv2.INTER_CUBIC`)
3. Apply **unsharp masking** to recover edge sharpness lost to interpolation:
   ```python
   blurred = cv2.GaussianBlur(upscaled, (5, 5), 0)
   amount = 0.5
   upscaled = cv2.addWeighted(upscaled, 1.0 + amount, blurred, -amount, 0)
   ```

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Target minimum width | 6000 px |
| Interpolation | `INTER_CUBIC` |
| Unsharp mask sigma | 5×5 Gaussian, `amount = 0.5` |

---

### 1.2 Petri Dish / ROI Detection — Hough Circle Transform

**Purpose:** Locate the circular petri dish in the image to crop and mask everything outside the dish boundary.

Three detection strategies are attempted in order:

#### Strategy 1: Hardcoded Lookup (Known Demo Images)
For known demo image filenames, exact circle parameters (cx, cy, r) are hardcoded in processed/upscaled coordinate space:
```python
HARDCODED_CIRCLES_PROCESSED = {
    "WIN_20250905_11_49_20_Pro.jpg": (3047, 2247, 1629),
    "complex 1.jpg": (3000, 3650, 1500),
    "85.bmp": (2872, 2437, 1261),
    ...
}
```

#### Strategy 2: MaskROI-like Hough (Primary Auto-Detection)
Replicates IncuCount's `MaskROI.cpp` detection logic:

1. **Internal resize for detection** — downscale to max 1200 px (or upscale small images ≤600 px to ~800 px, capped at 1.5×)
2. **Attempt 1 — Grayscale Hough:**
   ```python
   resized_gray = cv2.cvtColor(resized_color, cv2.COLOR_BGR2GRAY)
   blurred_gray = cv2.GaussianBlur(resized_gray, (15, 15), 0)
   circles = cv2.HoughCircles(blurred_gray, cv2.HOUGH_GRADIENT,
       dp=1.0, minDist=min_dist,
       param1=75, param2=35,
       minRadius=min_radius, maxRadius=max_radius)
   ```
3. **Attempt 2 — HSV S-Channel Fallback** (if grayscale fails):
   ```python
   hsv = cv2.cvtColor(resized_color, cv2.COLOR_BGR2HSV)
   s_channel = hsv[:, :, 1]
   blurred_s = cv2.GaussianBlur(s_channel, (15, 15), 0)
   circles = cv2.HoughCircles(blurred_s, cv2.HOUGH_GRADIENT,
       dp=1.0, minDist=min_dist,
       param1=50, param2=30, ...)
   ```
4. **Best circle selection** — choose the candidate circle closest to the image centre (Euclidean distance of centre vs. image midpoint)

**Hough parameters derived dynamically:**
| Parameter | Formula |
|-----------|---------|
| `min_radius` | `int(min_dim × 0.20)` |
| `max_radius` | `max(int(min_dim × 0.5), min_radius + 1)` |
| `min_dist` | `float(min_dim) × 0.5` |

#### Strategy 3: Legacy 256-px Hough (Final Fallback)
Replicates the original OpenCFU `MaskROI.cpp` circle detection:
```python
# Resize gray image to 256 px width
r = 256.0 / float(w)
resized = cv2.resize(gray, (0,0), fx=r, fy=r, interpolation=cv2.INTER_AREA)
resized = cv2.medianBlur(resized, 7)
circles = cv2.HoughCircles(resized, cv2.HOUGH_GRADIENT,
    dp=2.0, minDist=100.0,
    param1=150.0, param2=10.0,
    minRadius=75, maxRadius=350)
```

---

### 1.3 Circular Crop & Background Masking

After circle detection:

1. **Bounding box crop** — compute the tight bounding rectangle of the circle mask and crop:
   ```python
   mask = np.zeros((h, w), dtype=np.uint8)
   cv2.circle(mask, (cx, cy), radius, 255, -1)
   x, y, bw, bh = cv2.boundingRect(cv2.findNonZero(mask))
   cropped = bgr[y:y+bh, x:x+bw]
   ```

2. **Background fill** — pixels outside the circle get replaced with the **mean colour sampled from a thin annulus** (8–25 px ring near the dish edge):
   ```python
   inner = max(1, r - 25)
   outer = max(inner + 1, r - 8)
   # sample ring_pixels from annulus
   mean = ring_pixels.mean(axis=0)
   bgr[inv_mask == 255] = mean_color
   ```
   This prevents OpenCFU from detecting false colonies in the image corners.

---

## 2. OpenCFU C++ Detection Pipeline Algorithms

### 2.1 Step 1 — Noise Reduction

**Algorithms:**
- **Median blur** (`cv::medianBlur`) — removes salt-and-pepper noise
- **Gaussian blur** (`cv::GaussianBlur`) — smooths the image

**Kernel size** is adaptive, derived from the minimum colony radius:
```cpp
int s = std::min(cols/3, rows/3);
if (s < min_rad) min_rad = s;
if (min_rad > 1)
    kernel_size = (((min_rad - 1) / 4) * 2) + 1;
```

---

### 2.2 Step 2 — Illumination Correction (Background Subtraction)

This is the most critical preprocessing step. It uses a **multi-scale background estimation** approach:

**Threshold modes:**
| Mode | Operation | Use Case |
|------|-----------|---------|
| `reg` (NORMAL) | `255*(bg/mask) - foreground` | Dark colonies on light background |
| `inv` (INVERTED) | `foreground - 255*(bg/mask)` | Light colonies on dark background |
| `bilat` (BILATERAL) | `abs(foreground - bg)` | Mixed contrast |

**Background estimation:**
1. Downsample the image to ~196 px width (for speed)
2. Apply a strong **median blur** (11×11) to remove colony signals
3. Upsample back to original size
4. Subtract normalised background from each colour channel independently

**Laplacian-of-Gaussian (LoG) enhancement:**
After background subtraction, a LoG edge-enhancement is applied per channel:
```cpp
cv::GaussianBlur(in, tmp, Size(blurSize, blurSize), 3);
cv::Laplacian(tmp, tmp, CV_8U, 5, 0.3);
// Remove filled holes from Laplacian
cv::findContours(tmp, contours, ...);
out = in - tmp_mat;
```

The final result is the **per-channel average**: `(R + G + B) / 3` → grayscale enhanced image.

---

### 2.3 Step 3 — Pass One: Multi-Threshold Contour Detection + Random Forest Classification

This is the primary candidate detection step. It generates **contour families** across multiple threshold levels, then classifies each with a trained Random Forest.

#### 3a. Multi-threshold contour extraction

```cpp
// Sweep threshold values from min to max pixel value in 2-step increments
for (unsigned int i = 2; i < (max - min); i += 2) {
    cv::threshold(src, thrd, i + min, 255, THRESH_BINARY);
    cv::findContours(thrd, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
}
```

Large contours (>100 points) are **subsampled to 100 points** using linear interpolation for speed.

#### 3b. Feature extraction (13 features per contour)

```
Feature 0: Compactness = perimeter² / area
Feature 1: Convexity  = (hull_area - area) / hull_area
Feature 2: Solidity   = (hull_perimeter - perimeter) / hull_perimeter
Feature 3: Hole area ratio = area_hole / total_area
Feature 4: Hole perimeter ratio = perimeter_hole / total_perimeter
Feature 5: Aspect ratio = width / (width + height)
Features 6–12: Hu Moments (7 invariant moments)
```

**Contour smoothing** is applied before feature extraction using a wrap-around linear blur kernel:
```cpp
int k = 2 * (n_points / 40) + 1;  // k bounded to [3, 99]
cv::copyMakeBorder(contour, padded, k/2, k/2, 0, 0, BORDER_WRAP);
cv::blur(padded, smoothed, Size(1, k));
```

#### 3c. Random Forest Prediction (Pass 1 Classifier)

- **Classifier file:** `core_engine/src/classifier/trainnedClassifier.xml` (2.6 MB)
- **Algorithm:** OpenCV `cv::ml::RTrees` (Random Forest)
- **Configuration:**
  ```
  MaxDepth:          10
  MinSampleCount:    10
  RegressionAccuracy: 0
  UseSurrogates:     false
  MaxCategories:     3
  CalculateVarImportance: true
  MaxTrees:          100 (termination criteria: 0.01 EPS)
  ```
- **Output:** Binary classification per contour — `'N'` (Not colony) or other (candidate colony)
- Contours classified as valid are drawn onto a binary mask for Step 4.

---

### 2.4 Step 4 — Pass Two: Threshold + Contour Splitting + Final Classification

#### 4a. Global threshold (configurable)
```cpp
// Fixed threshold
cv::threshold(src, tmp, m_threshold, 255, THRESH_BINARY);

// OR Otsu's auto-threshold
cv::threshold(src, tmp, 0, 255, THRESH_BINARY | THRESH_OTSU);
```

#### 4b. ContourSplitter — overlapping colony separation

Clusters of overlapping/touching colonies (detected by Pass 1 as multi-colony blobs) are **split** using `ContourSpliter.cpp`. This handles the common case where two adjacent colonies form a single contiguous blob.

#### 4c. Two-pass classification after splitting

Colonies are split into two groups:
- `contour_fams_split` — blobs that were split (≥2 per cluster)
- `contour_fams_unsplit` — single-colony blobs

Each group is independently re-classified using a **second trained Random Forest** (`TRAINED_CLASSIF_PS_XML_FILE`) with the same 13-feature vector. Label `'S'` = valid single colony.

---

### 2.5 Post-Processing Filters

| Step | Algorithm |
|------|-----------|
| `Step_FiltGUI` | Applies user-drawn ROI mask filter |
| `Step_FiltIPosition2D` | Removes colonies outside the detected circular ROI boundary |
| `Step_FiltHS` | Hue/saturation range filter (optional) |
| `Step_FiltLik` | Likelihood threshold filter (removes low-confidence detections) |
| `Step_ColourCluster` | DBSCAN colour clustering for colony type grouping |

---

### 2.6 DBSCAN Colour Clustering (`Step_ColourCluster`)

When `enable_color_grouping = true`, colonies are grouped by colour using a custom **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) implementation in **CIE L\*a\*b\* colour space**.

**Distance metric (CIE76 colour difference):**
```
ΔE = sqrt(0.153787*(ΔL)² + (Δa)² + (Δb)²)
```
(The `0.153787 = (100/255)²` scaling factor corrects for OpenCV's 8-bit L* encoding)

**DBSCAN parameters (user-configurable):**
| Parameter | Default | Range |
|-----------|---------|-------|
| `coarseness` (epsilon distance) | 10.0 | 0.1–50.0 |
| `neighbours` (min points) | 10 | 4–50 |

**Output:** Colony group number (`Colour_group` in CSV) — cluster 1 = most populous colony colour group.

---

## 3. Result Data Per Colony

OpenCFU outputs the following per detected colony:

| Field | Type | Description |
|-------|------|-------------|
| `IsValid` | int | 1 = valid colony |
| `X` | float | X centroid (pixels, in processed space) |
| `Y` | float | Y centroid |
| `ROI` | int | ROI region index |
| `Colour_group` | int | DBSCAN cluster ID (0 = unclustered) |
| `N_in_clust` | int | Colonies split from this blob |
| `Area` | float | Colony area in pixels² |
| `Radius` | float | Estimated radius in pixels |
| `Hue` | float | Mean hue (HSV, 0–360) |
| `Saturation` | float | Mean saturation (HSV, 0–255) |
| `Rmean/Gmean/Bmean` | float | Mean RGB colour |
| `Rsd/Gsd/Bsd` | float | Std dev of RGB |

The Node.js wrapper (`colonyDetector.js`) then **transforms coordinates back** to original image space by reversing the preprocessing scale factor and crop offsets.
