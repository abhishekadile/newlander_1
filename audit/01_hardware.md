# 01 — Hardware, Camera & Board

## 1. Server / Compute Platform

### Target Environment

| Attribute | Value |
|-----------|-------|
| **OS** | Debian GNU/Linux 11 "Bullseye" (Docker image: `node:20-bullseye`) |
| **Runtime** | Node.js **v20 LTS** (Express 4) |
| **Python** | Python **3** (`python3` binary, system package) |
| **C++ Engine** | OpenCFU (pre-compiled, installed via `apt-get opencfu`) |
| **Architecture** | x86-64 (amd64) — Windows `.exe` binary also bundled for dev |

### Docker Specification

```dockerfile
FROM node:20-bullseye

RUN apt-get update && apt-get install -y \
    opencfu \       # core colony counter
    python3 \
    python3-pip \
    libgl1 \        # OpenCV runtime dependency
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy opencv-python-headless

EXPOSE 3000
CMD ["node", "server.js"]
```

**Production deployment:** Designed for Render.com or any Docker-compatible cloud host.  
**Local development (Windows):** Uses the bundled `core_engine/opencfu.exe` binary.

---

## 2. Camera

> **No dedicated hardware camera is specified or required by the API itself.**  
> The API operates entirely on **image files** submitted via HTTP upload or referenced by a pre-loaded image ID.

### What the codebase tells us about cameras

#### A. Image names embedded in the server suggest a Windows webcam

The file `server.js` has a hardcoded image catalog:

```js
'WIN_20250905_11_49_20_Pro': 'WIN_20250905_11_49_20_Pro.jpg',
'WIN_20250905_11_48_18_Pro': 'WIN_20250905_11_48_18_Pro.jpg',
'WIN_20250905_11_42_42_Pro': 'WIN_20250905_11_42_42_Pro.jpg',
'WIN_20250905_11_44_26_Pro': 'WIN_20250905_11_44_26_Pro.jpg',
```

The prefix `WIN_` and timestamp format `YYYYMMDD_HH_MM_SS_Pro` is the **default file-naming scheme of the Windows Camera app** (also used by Surface devices). The suffix `_Pro` indicates a **Surface Pro** or similar Microsoft device.

This is strong evidence that the images were **captured with a Windows device webcam** (likely a **Microsoft Surface Pro** internal camera or a USB webcam connected to a Windows PC).

#### B. Colony Profile preset: camera control params

In `utils/colonyProfilePreset.js`, each colony detection profile stores camera metadata:

```js
const DEFAULT_CAMERA = {
  brightness: 0,
  exposure: 0,
  contrast: 0,
  device_label_hint: null,   // e.g. "USB Camera", "Integrated Webcam"
};
```

- `device_label_hint` is a free-text field, meaning the desktop app (client) identifies its own camera by label and stores that label in the profile.
- Camera control parameters (brightness, exposure, contrast) imply the desktop app can apply software-level camera adjustments before sending the image.

#### C. Lighting relay control

The preset schema includes:

```js
const DEFAULT_LIGHTING = {
  relays: [false, false, false, false, true, true, false, false],
};
```

This is an **8-channel relay board** controlling individual lights around the imaging enclosure. Channels 4 and 5 (`true`) are on by default, suggesting an overhead or side-lighting ring light arrangement. This indicates the IncuCount desktop unit is a **custom hardware enclosure** with controlled illumination.

---

## 3. Supported Image Formats & Dimensions

OpenCFU (via the C++ `setImage()` method in `ProcessingOptions.hpp`) uses:

```cpp
cv::Mat tmpImg = cv::imread(str, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);

// 16-bit depth patch:
if (tmpImg.depth() == CV_16U) {
    // scale down to 8-bit
}
```

| Format | Support |
|--------|---------|
| JPEG/JPG | ✅ Full |
| PNG | ✅ Full |
| BMP | ✅ Full |
| 16-bit TIFF/PNG | ✅ Auto-normalised to 8-bit |
| Any OpenCV-readable format | ✅ |

**Image upscaling target:** `preprocess.py` upscales any image narrower than **6000 px** to 6000 px width (bicubic + mild unsharp mask) before passing to OpenCFU. This is a critical parameter matching the OpenCFU C++ pipeline internal behaviour.

---

## 4. Board / Embedded Platform

The codebase does **not** use an embedded SBC (Raspberry Pi, Arduino, etc.) directly in this repository. The API is a **cloud/server process**. However:

- The **8-channel relay board** referenced in the lighting configuration strongly implies the IncuCount desktop unit pairs with a microcontroller or relay HAT (likely on a Raspberry Pi or similar).
- The `device_label_hint` camera field and relay array suggest a companion **desktop Electron app** on a local Windows/Mac machine that interfaces with hardware, then sends images to this cloud API.

### Inferred system topology

```
┌─────────────────────────────────────────┐
│  IncuCount Desktop Enclosure             │
│  ┌───────────┐   ┌────────────────────┐ │
│  │ USB Camera│   │ 8-ch Relay Board   │ │
│  │ (webcam)  │   │ (ring lights etc.) │ │
│  └─────┬─────┘   └─────────┬──────────┘ │
│        │                   │            │
│  ┌─────▼───────────────────▼──────────┐ │
│  │    Windows / Mac PC (Electron App)  │ │
│  │  - Camera capture                   │ │
│  │  - Relay control                    │ │
│  │  - Image upload → IncuCountAPI      │ │
│  └────────────────────────────────────┘ │
└────────────────┬────────────────────────┘
                 │ HTTPS POST /detect
                 ▼
       ┌──────────────────┐
       │ IncuCountAPI      │
       │ (Docker / Render) │
       │ Node.js + OpenCFU │
       └──────────────────┘
```

---

## 5. Key Hardware-Related Parameters (Summary)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lighting.relays[4]` | `true` | Light relay channel 5 ON by default |
| `lighting.relays[5]` | `true` | Light relay channel 6 ON by default |
| `camera.brightness` | `0` | Webcam brightness offset |
| `camera.exposure` | `0` | Webcam exposure offset |
| `camera.contrast` | `0` | Webcam contrast offset |
| `camera.device_label_hint` | `null` | Camera device label |
| Min image width for full accuracy | `6000 px` | Enforced by preprocessing upscale |
| Supported colour depth | 8-bit and 16-bit (auto-converted) | C++ OpenCV `IMREAD_ANYDEPTH` |
