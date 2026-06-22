# 03 — Pipeline

## End-to-End Processing Pipeline

```
Client (HTTP POST /detect)
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  1. API Layer (server.js)                             │
│     • Accept multipart image upload OR image_id ref   │
│     • Parse detection params from request body        │
│     • Validate path traversal safety                  │
└──────────────────────┬────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────┐
│  2. ColonyDetector.detectColonies() (colonyDetector.js)│
│     • Validate image file existence                   │
│     • Read image dimensions via Sharp                 │
└──────────────────────┬────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────┐
│  3. Python Preprocessing (preprocess.py)              │
│     Spawned as: python3 preprocess.py --image <path>  │
│                                                       │
│     a) Upscale to min 6000 px width (bicubic + USM)  │
│     b) Detect Petri dish circle (Hough or hardcoded) │
│     c) Crop to circle bounding box                   │
│     d) Mask background outside circle                │
│     e) Save processed PNG to temp file               │
│     f) Return JSON metadata to stdout                 │
└──────────────────────┬────────────────────────────────┘
                       │  (JSON via stdout)
                       ▼
┌───────────────────────────────────────────────────────┐
│  4. OpenCFU C++ Engine (core_engine/opencfu.exe)      │
│     Spawned as: opencfu -i <path> -d reg -t 15 ...    │
│                                                       │
│     Step 1: Noise Reduction (median + Gaussian blur)  │
│     Step 2: Illumination Correction (bg subtraction   │
│             + LoG enhancement)                        │
│     Step 3: Multi-threshold Contour + RF Pass 1       │
│     Step 4: Global Threshold + Splitting + RF Pass 2  │
│     Filter: GUI mask, Position, Hue/Sat, Likelihood   │
│     Filter: DBSCAN Colour Clustering (optional)       │
│     Output: bacterial_colonies.csv                    │
└──────────────────────┬────────────────────────────────┘
                       │  (CSV file)
                       ▼
┌───────────────────────────────────────────────────────┐
│  5. Result Parsing & Coordinate Transform             │
│     (colonyDetector.js)                               │
│                                                       │
│     a) Parse CSV → colony array                       │
│     b) Transform (x,y,r) from processed space         │
│        back to original image coordinates             │
│        x_orig = (x_proc + offset_x) / scale_factor   │
│     c) Filter colonies outside detected dish circle   │
└──────────────────────┬────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────┐
│  6. JSON Response                                     │
│     { success, colonies[], colonyCount,               │
│       imageDimensions, detectedCircle, meta }         │
└───────────────────────────────────────────────────────┘
```

---

## Stage Details

### Stage 1 — HTTP API Layer (`server.js`)

**Endpoint:** `POST /detect`  
**Content types accepted:**
- `multipart/form-data` — file upload via `multer` (stored in `/uploads/`)
- `application/json` — body with `image_id` referencing pre-loaded server images

**Request parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | file | — | Uploaded image file |
| `image_id` | string | — | ID for server-side image |
| `threshold_type` | string | `'regular'` | `'regular'` or `'inverted'` |
| `threshold_value` | number | `15` | Detection threshold (0–255) |
| `min_radius` | number | `3` | Min colony radius (px) |
| `max_radius` | number | `50` | Max colony radius (px) |
| `enable_color_grouping` | bool | `false` | Enable DBSCAN colour clustering |
| `coarseness` | number | `10.0` | DBSCAN epsilon distance |
| `neighbours` | number | `10` | DBSCAN min points |

**File storage:** Multer stores uploaded files with a unique timestamped name:
```js
filename: `image-${Date.now()}-${Math.round(Math.random() * 1E9)}.ext`
```

> ⚠️ **Issue:** Uploaded files are **never deleted** after processing. Over time this will exhaust disk space.

---

### Stage 2 — ColonyDetector Orchestration (`colonyDetector.js`)

The `ColonyDetector` class is the central orchestrator:

1. **Binary resolution:**
   - Windows dev: `core_engine/opencfu.exe`
   - Linux/Docker: `opencfu` (system PATH)
   - Override: `OPENCFU_BIN` env variable

2. **Sharp metadata:** Reads image dimensions (`width`, `height`) using the Sharp library for frontend display. This happens even if preprocessing is skipped.

3. **Stale result prevention:** Before running OpenCFU, the existing `bacterial_colonies.csv` is **deleted** to prevent serving cached results from a previous run:
   ```js
   await fs.remove(this.csvOutputPath);
   ```

---

### Stage 3 — Python Preprocessing (`preprocess.py`)

**Communication:** Spawned as a subprocess; result is **JSON on stdout**.

**Environment variables:**
| Variable | Default | Purpose |
|----------|---------|---------|
| `PYTHON_BIN` | `python` (Win) / `python3` (Linux) | Python binary |

**Output JSON schema:**
```json
{
  "processed_image_path": "/tmp/.../preprocessed_abc123.png",
  "created_temp_file": true,
  "cropped": true,
  "scale_factor": 2.1,
  "crop_offset_processed": {"x": 418, "y": 318},
  "crop_offset": {"x": 199, "y": 151},
  "original_size": {"width": 2856, "height": 2142},
  "processed_size": {"width": 3258, "height": 3258},
  "upscaled_size": {"width": 6000, "height": 4500},
  "detected_circle_processed": {
    "present": true,
    "source": "auto",
    "center": {"x": 3047, "y": 2247},
    "radius": 1629
  },
  "detected_circle": {
    "present": true,
    "source": "hardcoded",
    "center": {"x": 1451, "y": 1070},
    "radius": 775
  },
  "debug": { ... }
}
```

**Detection source hierarchy:**

| Priority | Source | Condition |
|---------|--------|-----------|
| 1 | `hardcoded_processed` | Image basename matches `HARDCODED_CIRCLES_PROCESSED` dict |
| 2 | `upscaled` | Hough on upscaled image succeeds |
| 3 | `original_fallback` | Hough on upscaled fails; try on original image |
| 4 | `legacy256_original_fallback` | All Hough attempts fail; use legacy 256-px method |

**Temp file location:** `<script_dir>/temp_preprocessing/preprocessed_<uuid>.png`

---

### Stage 4 — OpenCFU C++ Engine

**Command line construction:**
```js
const args = [
  '-i', path.resolve(imagePath),     // input image
  '-d', threshold_type === 'inverted' ? 'inv' : 'reg',  // direction
  '-t', String(threshold_value),     // threshold
  '-r', String(min_radius),          // min radius
  '-R', String(max_radius),          // max radius
];

if (enable_color_grouping) {
  args.push(`-D${coarseness}`);     // DBSCAN epsilon
  args.push(`-N${neighbours}`);     // DBSCAN min points
}
```

**CWD:** `core_engine/` — OpenCFU writes `bacterial_colonies.csv` to its working directory.

**OpenCFU internal step sequence:**

```
Constructor:
  Load trainnedClassifier.xml          → Predictor (Pass 1 RF)
  Load trainnedClassifier_ps.xml       → Predictor_ps (Pass 2 RF)

runAll():
  Step_1    → Noise reduction
  Step_2    → Illumination correction
  Step_3    → Multi-thresh contours + Pass 1 RF → binary mask
  Step_4    → Global threshold + splitting + Pass 2 RF → Result
  Step_FiltGUI    → GUI mask filter
  Step_FiltIPosition2D → Circle boundary filter
  Step_FiltHS     → Hue/Saturation filter
  Step_FiltLik    → Likelihood filter
  Step_ColourCluster → DBSCAN colour grouping

writeResult():
  Print CSV header + rows to stdout
  Save to bacterial_colonies.csv
```

**OMP parallelism:** Steps 2 and 3 use `#pragma omp parallel for` — multi-threaded within each step.

---

### Stage 5 — Coordinate Transformation

OpenCFU operates in **processed (upscaled + cropped) coordinate space**. Before returning results, coordinates are mapped back to **original image space**:

```
x_original = (x_processed + crop_offset_x_processed) / scale_factor
y_original = (y_processed + crop_offset_y_processed) / scale_factor
r_original = r_processed / scale_factor
area_original = area_processed / (scale_factor²)
```

**Post-transformation filter:** Any colony whose centre is more than `1.02 × dish_radius` from the dish centre (in original coordinates) is discarded:
```js
const tol = 1.02;
const r2 = (r * tol) * (r * tol);
const dx = x - cx, dy = y - cy;
keep = (dx*dx + dy*dy) <= r2;
```

---

### Stage 6 — API Response

**Success response shape:**
```json
{
  "success": true,
  "colonies": [
    {
      "isvalid": "1",
      "x": 1451.5,
      "y": 1660.5,
      "roi": "1",
      "colour_group": "0",
      "n_in_clust": "1",
      "area": 234.7,
      "radius": 8.6,
      "hue": 44,
      "saturation": 10,
      "rmean": 136.9, "gmean": 134.7, "bmean": 129.5,
      "rsd": 10.9, "gsd": 11.3, "bsd": 11.6
    },
    ...
  ],
  "colonyCount": 128,
  "imageDimensions": {"width": 2856, "height": 2142},
  "processedImagePath": "uploads/image-...-123.png",
  "originalImagePath": "uploads/image-...-456.jpg",
  "meta": { "cropped": true, "scale_factor": 2.1, ... },
  "resultSource": "csv",
  "detectedCircle": {"present": true, "center": {...}, "radius": 775}
}
```

---

## Other API Routes

| Route | Method | Auth | Description |
|-------|--------|------|-------------|
| `/detect` | POST | None | Colony detection |
| `/health` | GET | None | Health check |
| `/images/catalog` | GET | None | List known server images |
| `/api/auth/register` | POST | None | User registration |
| `/api/auth/login` | POST | None | User login (JWT) |
| `/api/auth/me` | GET | Bearer JWT | Get current user |
| `/api/auth/logout` | POST | None | Logout (stateless) |
| `/api/license/seed` | POST | Admin secret | Seed license keys |
| `/api/license/register` | POST | None | Register device to license |
| `/api/license/check` | POST | None | Check device license |
| `/api/license/list` | GET | Admin secret | List all licenses |
| `/api/license/:key` | DELETE | Admin secret | Remove license |
| `/api/colony-profiles` | GET | None | List colony profiles |
| `/api/colony-profiles` | POST | None | Create profile |
| `/api/colony-profiles/:id` | PUT | Bearer JWT | Update profile |
| `/api/colony-profiles/:id` | DELETE | Bearer JWT | Delete profile |
| `/api/colony-profiles/admin/list` | GET | Admin secret | Admin list all profiles |
| `/api/colony-profiles/admin/:id/validation` | PATCH | Admin secret | Approve/revoke profile |
| `/api/colony-profiles/admin/:id` | DELETE | Admin secret | Admin delete profile |
| `/api/colony-profiles/seed` | POST | Admin secret | Seed default profiles |
| `/api/logs` | POST | None (rate-limited) | Ingest diagnostic logs |
| `/api/download/...` | GET | License token | Download installer |

---

## Colony Profile System

Colony profiles store pre-configured detection parameters for specific plate types:

| Profile | threshold_type | threshold_value | min_radius | max_radius | color_grouping |
|---------|----------------|-----------------|------------|------------|----------------|
| Anaerobic Count Film | regular | 15 | 3 | 50 | No |
| Coliform (CC) Film | regular | 15 | 3 | 50 | No |
| MacConkey Plates | regular | 15 | 3 | 50 | **Yes** |
| Nutrient Plates | regular | 5 | 4 | **185** | No |

Profiles follow a versioned schema (`schema_version: 1`) with sections for `detection`, `camera`, `lighting`, `region`, `workflow`, and `locks`.
