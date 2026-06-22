# 05 — Latency

## Overview

The system has **three main latency contributors** that run sequentially per request:

```
Total latency = API overhead + Python preprocessing + OpenCFU detection
```

All three stages are **single-threaded from the request perspective** (spawned sequentially), though OpenCFU internally parallelises some steps with OpenMP.

---

## 1. Per-Stage Latency Breakdown

### Stage 1: API Overhead (Node.js)

| Sub-step | Estimated Latency |
|----------|-----------------|
| Multer file save to disk | 10–50 ms |
| Sharp image metadata read | 5–20 ms |
| Parameter parsing & validation | <1 ms |
| Response serialisation | 1–5 ms |
| **Stage total** | **~16–75 ms** |

**Bottleneck:** Disk I/O for file save (depends on storage speed).

---

### Stage 2: Python Preprocessing (`preprocess.py`)

The preprocessing latency depends heavily on **image resolution** and **whether upscaling is needed**.

#### Sub-step breakdown:

| Sub-step | Small image (≤3MP) | Large image (≥8MP) |
|----------|-------------------|-------------------|
| Process startup (Python init) | 200–500 ms | 200–500 ms |
| Image read (`cv2.imread`) | 50–200 ms | 200–800 ms |
| Upscale to 6000 px (bicubic) | 500–2000 ms | 50–200 ms (already large) |
| Unsharp mask | 100–400 ms | 100–400 ms |
| Hough circle detection (multi-scale) | 100–300 ms | 100–300 ms |
| Crop + background mask | 50–150 ms | 100–300 ms |
| `cv2.imwrite` PNG to temp | 500–2000 ms | 500–2000 ms |
| JSON stdout output | <5 ms | <5 ms |
| **Stage total (cold)** | **~1.5–5.5 sec** | **~1.3–4.5 sec** |

> ⚠️ **Critical bottleneck:** The Python process startup takes 200–500 ms on every request because `cv2` and NumPy are imported fresh. There is no process pool or warm-start mechanism.

> ⚠️ **PNG write is slow:** Writing a 6000-px PNG is CPU-intensive (LZ77 compression). Switching to JPEG (quality 95) could reduce this to 200–800 ms.

---

### Stage 3: OpenCFU Detection

OpenCFU latency depends on image size (after preprocessing → typically ~3000×3000 px crop of the dish) and colony density.

#### Internal step timing estimates (on a 3000×3000 processed image):

| Step | Algorithm | Estimated Time |
|------|-----------|---------------|
| Step 1 | Median blur + Gaussian blur | 50–200 ms |
| Step 2 | Background subtraction (196-px downsample, LoG) | 300–800 ms |
| Step 3 | Multi-threshold contour sweep (0→255, step 2) | **1–5 sec** |
| Step 4 | Global threshold + contour splitting | 200–800 ms |
| Step_FiltGUI | ROI mask filter | 10–50 ms |
| Step_FiltIPosition2D | Circle boundary filter | 10–50 ms |
| Step_FiltHS | Hue/saturation filter | 10–30 ms |
| Step_FiltLik | Likelihood filter | 5–20 ms |
| Step_ColourCluster | DBSCAN (if enabled, O(n²) per iteration) | 10–200 ms |
| CSV write | File write | 5–20 ms |
| **Stage total** | | **~2–7 sec** |

> ⚠️ **Step 3 is the dominant cost:** The multi-threshold contour sweep at step increments of 2 means ~127 threshold levels, each requiring a full `cv::threshold` + `cv::findContours` on a large image. This is the **primary latency driver** for OpenCFU.

> ℹ️ **OMP parallelism:** Step 2 and Step 3 use `#pragma omp parallel for`. On a 4-core machine, Step 3 can complete in ~1–2 sec instead of 4–6 sec.

---

## 2. Total End-to-End Latency Estimates

| Scenario | Preprocessing | OpenCFU | API | **Total** |
|---------|--------------|---------|-----|-----------|
| Small image (<1MP), few colonies | 1.5 sec | 2 sec | 0.1 sec | **~3.6 sec** |
| Medium image (4–8MP), typical | 3 sec | 3–4 sec | 0.1 sec | **~6–7 sec** |
| Large image (12MP+), dense plate | 4–5 sec | 5–7 sec | 0.1 sec | **~9–12 sec** |
| Known image (hardcoded circle) | 2–3 sec | 3–4 sec | 0.1 sec | **~5–7 sec** |
| With colour grouping (DBSCAN) | +0 | +0.1–0.2 sec | +0 | minimal impact |

**Typical real-world latency: ~5–10 seconds per image.**

---

## 3. Concurrency

### Current limitation

The server runs as a **single Node.js process** with **no request queuing or limiting** on the `/detect` endpoint. Multiple simultaneous requests will:

1. Spawn multiple Python preprocessing processes in parallel (CPU-intensive)
2. Attempt to run multiple OpenCFU processes concurrently
3. **Race condition on `bacterial_colonies.csv`** — all OpenCFU processes write to the same fixed filename in `core_engine/`. If two requests finish within milliseconds of each other, one may read the other's CSV.

> 🔴 **This is a critical concurrency bug for any multi-user deployment.**

### Rate limiting

The `/api/logs` route has a rate limiter (`logIngestRateLimit.js`), but **`/detect` has no rate limiting** at all.

---

## 4. Latency Hotspots Summary

| Rank | Bottleneck | Est. Impact | Fix |
|------|-----------|-------------|-----|
| 1 | Step 3 multi-threshold contour sweep | 1–5 sec | Reduce image size or use INTER_AREA downscale before OpenCFU |
| 2 | Python process startup (cold start) | 200–500 ms per request | Use a persistent Python process (e.g., Flask microservice) |
| 3 | PNG output in preprocessing | 500–2000 ms | Switch temp format to JPEG or BMP |
| 4 | Image upscaling to 6000px | 500–2000 ms | Make target resolution configurable or cap lower |
| 5 | No concurrency protection on CSV | Race condition | Add UUID to output CSV filename per request |

---

## 5. Latency vs. Parameter Impact

| Parameter | Impact on Latency |
|-----------|-----------------|
| `min_radius` ↑ | Fewer contours → slightly faster Step 3 |
| `max_radius` ↑ | More contours → slightly slower Step 3 |
| `threshold_value` | Affects Step 4 only — minimal impact |
| `enable_color_grouping = true` | Adds DBSCAN pass: +10–200ms depending on colony count |
| `coarseness` ↓ (smaller ε) | Smaller clusters → DBSCAN faster but more clusters |
| `neighbours` ↑ | DBSCAN O(n²) per query → slower for dense plates |
| Image resolution ↑ | Linear increase in preprocessing; polynomial in Step 3 |

---

## 6. Recommendations for Latency Reduction

### Quick wins (implementation effort: low)

1. **Switch preprocessing temp output to BMP or JPEG** instead of PNG to eliminate compression latency
   ```python
   out_path = f"preprocessed_{uuid}.bmp"  # ~5× faster write
   ```

2. **Use a unique output CSV path per request** to fix the race condition and enable parallel processing:
   ```js
   const csvOutputPath = path.join(coreEnginePath, `result_${uuid}.csv`);
   args.push('--output', csvOutputPath);  // if OpenCFU supports it
   ```
   *(Note: OpenCFU may need patching to support custom output paths)*

3. **Add rate limiting to `/detect`** to prevent CPU starvation under load

### Medium effort

4. **Persistent Python process** — replace `child_process.spawn` with a long-running FastAPI/Flask microservice that keeps OpenCV loaded in memory. This eliminates the 200–500ms startup tax per request.

5. **Configurable upscale target** — 6000px is aggressive; many use cases would be well-served by 3000–4000px with minimal accuracy loss and 4× speed improvement in both upscaling and OpenCFU processing.

### High effort

6. **OpenCFU image prescaling** — pass a pre-resized image to OpenCFU at a target of 2000–3000px. The current pipeline upscales to 6000px for OpenCFU, but OpenCFU then does internal operations at that resolution. Reducing to 3000px would cut OpenCFU time by ~4× (area scales quadratically).

7. **Request queue with job IDs** — move `/detect` to an async job model (return job ID immediately, poll for results) to support concurrent multi-user workloads.
