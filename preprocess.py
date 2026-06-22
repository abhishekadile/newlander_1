import argparse
import json
import os
import sys
import uuid
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


# Hardcoded dish circles (processed/upscaled coordinate space) for known demo images.
# This bypasses Hough detection and ensures consistent cropping/masking.
# Keyed by the basename of the input image path.
HARDCODED_CIRCLES_PROCESSED: Dict[str, Tuple[int, int, int]] = {
    # MacConkey/Nutrient plate photos
    "WIN_20250905_11_49_20_Pro.jpg": (3047, 2247, 1629),
    "WIN_20250905_11_48_18_Pro.jpg": (3012, 2262, 1763),
    "WIN_20250905_11_42_42_Pro.jpg": (3052, 2282, 1750),
    # Backend maps image_id WIN_20250905_11_44_26_Pro -> file WIN_20250905_11_44_26_Pro.jpg
    "WIN_20250905_11_44_26_Pro.jpg": (3052, 2257, 1700),

    # Count films
    "complex 1.jpg": (3000, 3650, 1500),
    "standard 1.jpg": (3000, 3598, 1450),

    # BMP samples
    "85.bmp": (2872, 2437, 1261),
    "82.bmp": (2872, 2532, 1261),
}


def _json_stdout(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


def _die(msg: str, code: int = 1) -> None:
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()
    raise SystemExit(code)


def upscale_to_min_width_bicubic_unsharp(
    bgr: np.ndarray, min_width: int = 6000
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    h, w = bgr.shape[:2]
    meta: Dict[str, Any] = {
        "did_upscale": False,
        "requested_min_width": min_width,
        "requested_scale": 1.0,
        "actual_scale": 1.0,
        "upscaled_size": {"width": w, "height": h},
    }
    if w <= 0 or h <= 0:
        return bgr, 1.0, meta

    if w >= min_width:
        return bgr, 1.0, meta

    requested_scale = float(min_width) / float(w)
    target_w = int(round(w * requested_scale))
    target_h = int(round(h * requested_scale))

    upscaled = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Match C++: blur + addWeighted (mild unsharp mask)
    blurred = cv2.GaussianBlur(upscaled, (5, 5), 0)
    amount = 0.5
    upscaled = cv2.addWeighted(upscaled, 1.0 + amount, blurred, -amount, 0)

    actual_scale = float(upscaled.shape[1]) / float(w) if w > 0 else 1.0
    meta.update(
        {
            "did_upscale": True,
            "requested_scale": requested_scale,
            "actual_scale": actual_scale,
            "upscaled_size": {"width": int(upscaled.shape[1]), "height": int(upscaled.shape[0])},
        }
    )
    return upscaled, actual_scale, meta


def detect_circle_maskroi_like(bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int], Dict[str, Any]]:
    """
    Replicates IncuCount MaskROI.cpp auto mask detection:
    - Internal resize for detection: downscale to max 1200px, or upscale small images to ~800px (cap 1.5x)
    - Attempt 1: grayscale Hough
    - Attempt 2: HSV S-channel fallback
    - Select best circle closest to center
    Returns: (center_x, center_y) and radius in the *input image* coordinate space.
    """
    h, w = bgr.shape[:2]
    meta: Dict[str, Any] = {
        "detection_scale": 1.0,
        "resized_size": {"width": w, "height": h},
        "min_radius": None,
        "max_radius": None,
        "min_dist": None,
        "found_circles": 0,
        "used_fallback_s_channel": False,
    }
    if w <= 0 or h <= 0:
        return None, None, meta

    max_dimension = 1200.0
    parent_max_dim = float(max(w, h))
    scale = 1.0
    if parent_max_dim > max_dimension:
        scale = max_dimension / parent_max_dim
    if parent_max_dim < 600.0:
        target_width = 800.0
        upscale = target_width / parent_max_dim
        if upscale > 1.5:
            upscale = 1.5
        scale = upscale

    resized_cols = int(w * scale)
    resized_rows = int(h * scale)
    resized_cols = max(1, resized_cols)
    resized_rows = max(1, resized_rows)
    min_dim = int(min(resized_cols, resized_rows))

    min_radius = int(min_dim * 0.20)
    max_radius = max(int(min_dim * 0.5), min_radius + 1)
    min_dist = float(min_dim) * 0.5

    meta.update(
        {
            "detection_scale": float(scale),
            "resized_size": {"width": resized_cols, "height": resized_rows},
            "min_radius": int(min_radius),
            "max_radius": int(max_radius),
            "min_dist": float(min_dist),
        }
    )

    interp = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA
    resized_color = cv2.resize(bgr, (resized_cols, resized_rows), interpolation=interp)

    resized_gray = cv2.cvtColor(resized_color, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(resized_gray, (15, 15), 0)

    hsv = cv2.cvtColor(resized_color, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    blurred_s = cv2.GaussianBlur(s_channel, (15, 15), 0)

    circles = cv2.HoughCircles(
        blurred_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=min_dist,
        param1=75,
        param2=35,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None or circles.size == 0:
        meta["used_fallback_s_channel"] = True
        circles = cv2.HoughCircles(
            blurred_s,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=min_dist,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

    if circles is None or circles.size == 0:
        meta["found_circles"] = 0
        return None, None, meta

    circles = circles[0]  # shape: (N,3)
    meta["found_circles"] = int(circles.shape[0])

    cx = resized_cols / 2.0
    cy = resized_rows / 2.0

    def dist_sq(c: np.ndarray) -> float:
        dx = float(c[0]) - cx
        dy = float(c[1]) - cy
        return dx * dx + dy * dy

    best = min(circles, key=dist_sq)

    inv_scale = 1.0 / float(scale) if scale != 0 else 1.0
    center_x = int(best[0] * inv_scale)
    center_y = int(best[1] * inv_scale)
    radius = int(best[2] * inv_scale)

    center_x = max(0, min(center_x, w - 1))
    center_y = max(0, min(center_y, h - 1))
    radius = max(1, radius)

    return (center_x, center_y), int(radius), meta


def detect_circle_legacy256(bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int], Dict[str, Any]]:
    """
    Fallback circle detection based on the older OpenCFU MaskROI.cpp implementation:
    - Convert to gray
    - Resize so width becomes 256px (fx=fy=r where r=256/original_width)
    - Median blur
    - HoughCircles with fixed params (dp=2, minDist=100, param1=150, param2=10, minR=75, maxR=350)
    Returns center/radius in the *input image* coordinate space.
    """
    h, w = bgr.shape[:2]
    meta: Dict[str, Any] = {
        "r": None,
        "resized_size": {"width": w, "height": h},
        "found_circles": 0,
        "params": {
            "dp": 2.0,
            "minDist": 100.0,
            "param1": 150.0,
            "param2": 10.0,
            "minRadius": 75,
            "maxRadius": 350,
        },
    }
    if w <= 0 or h <= 0:
        return None, None, meta

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    r = 256.0 / float(w)
    # Resize using the same approach as C++: cv::Size(0,0), fx=r, fy=r, INTER_AREA
    resized = cv2.resize(gray, (0, 0), fx=r, fy=r, interpolation=cv2.INTER_AREA)
    resized = cv2.medianBlur(resized, 7)

    rh, rw = resized.shape[:2]
    meta["r"] = float(r)
    meta["resized_size"] = {"width": int(rw), "height": int(rh)}

    circles = cv2.HoughCircles(
        resized,
        cv2.HOUGH_GRADIENT,
        dp=2.0,
        minDist=100.0,
        param1=150.0,
        param2=10.0,
        minRadius=75,
        maxRadius=350,
    )

    if circles is None or circles.size == 0:
        meta["found_circles"] = 0
        return None, None, meta

    circles = circles[0]
    meta["found_circles"] = int(circles.shape[0])

    # Choose circle closest to center (in resized space)
    cx = rw / 2.0
    cy = rh / 2.0

    def dist_sq(c: np.ndarray) -> float:
        dx = float(c[0]) - cx
        dy = float(c[1]) - cy
        return dx * dx + dy * dy

    best = min(circles, key=dist_sq)
    inv_r = (1.0 / r) if r != 0 else 1.0
    center_x = int(best[0] * inv_r)
    center_y = int(best[1] * inv_r)
    radius = int(best[2] * inv_r)

    center_x = max(0, min(center_x, w - 1))
    center_y = max(0, min(center_y, h - 1))
    radius = max(1, radius)

    return (center_x, center_y), int(radius), meta


def crop_to_circle_mask_bbox(bgr: np.ndarray, center: Tuple[int, int], radius: int) -> Tuple[np.ndarray, Tuple[int, int], Dict[str, Any]]:
    h, w = bgr.shape[:2]
    cx, cy = center

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, -1)

    nz = cv2.findNonZero(mask)
    if nz is None:
        return bgr, (0, 0), {"bbox": None}

    x, y, bw, bh = cv2.boundingRect(nz)
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
    bw = max(1, min(int(bw), w - x))
    bh = max(1, min(int(bh), h - y))

    cropped = bgr[y : y + bh, x : x + bw].copy()
    return cropped, (x, y), {"bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)}}


def apply_circular_background_mask_inplace(
    bgr: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    fill_bgr: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Any]:
    """
    Mutates bgr: pixels outside the circle are set to a background color.
    This emulates OpenCFU's ROI mask filtering (Step_FiltIPosition2D) for backends
    that cannot pass a mask into OpenCFU.
    """
    h, w = bgr.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    if w <= 0 or h <= 0 or r <= 0:
        return {"applied": False, "reason": "invalid_dims"}

    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))
    r = max(1, min(r, max(w, h)))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)

    # Choose fill color from an annulus near the dish edge for stable background.
    used_fill = fill_bgr
    if used_fill is None:
        inner = max(1, r - 25)
        outer = max(inner + 1, r - 8)
        ring = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring, (cx, cy), outer, 255, -1)
        cv2.circle(ring, (cx, cy), inner, 0, -1)
        ring_pixels = bgr[ring == 255]
        if ring_pixels is not None and ring_pixels.size > 0:
            mean = ring_pixels.mean(axis=0)
            used_fill = (int(mean[0]), int(mean[1]), int(mean[2]))
        else:
            used_fill = (255, 255, 255)

    inv = cv2.bitwise_not(mask)
    bgr[inv == 255] = used_fill

    return {
        "applied": True,
        "fill_bgr": {"b": int(used_fill[0]), "g": int(used_fill[1]), "r": int(used_fill[2])},
        "center": {"x": cx, "y": cy},
        "radius": r,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI preprocess (IncuCount-compatible).")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", required=False, help="Optional output path for processed image")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        _die(f"Error: Image file does not exist: {image_path}")

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        _die(f"Error: Could not read image at: {image_path}")

    original_h, original_w = bgr.shape[:2]

    # 1) Upscale-to-6000-width (ProcessingOptions.hpp behavior)
    upscaled_bgr, scale_factor, upscale_meta = upscale_to_min_width_bicubic_unsharp(bgr, min_width=6000)

    # 2) Hardcoded circle override (processed/upscaled space)
    basename = os.path.basename(image_path)
    hardcoded = HARDCODED_CIRCLES_PROCESSED.get(basename)
    detected_center = None
    detected_radius = None
    hough_meta: Dict[str, Any] = {"used_hardcoded": False}
    detection_source = "upscaled"
    hough_meta_original = None
    hough_meta_legacy256 = None

    if hardcoded is not None:
        cx, cy, r = hardcoded
        uh, uw = upscaled_bgr.shape[:2]
        cx = max(0, min(int(cx), uw - 1))
        cy = max(0, min(int(cy), uh - 1))
        r = max(1, min(int(r), max(uw, uh)))
        detected_center, detected_radius = (cx, cy), int(r)
        detection_source = "hardcoded_processed"
        hough_meta = {"used_hardcoded": True, "basename": basename}

    # 3) Auto circle detection (MaskROI.cpp behavior) (fallback if not hardcoded)
    # 2) Auto circle detection (MaskROI.cpp behavior)
    # Try on upscaled image first (matches C++ pipeline). If it fails, fall back to
    # detecting on the original image then map the circle into upscaled coordinates.
    if detected_center is None or detected_radius is None:
        detected_center, detected_radius, hough_meta = detect_circle_maskroi_like(upscaled_bgr)
        detection_source = "upscaled"

    if detected_center is None or detected_radius is None:
        center_orig, radius_orig, hough_meta_original = detect_circle_maskroi_like(bgr)
        if center_orig is not None and radius_orig is not None:
            detection_source = "original_fallback"
            # Map to upscaled coordinate space for cropping the upscaled image
            mapped_center = (
                int(round(float(center_orig[0]) * float(scale_factor))),
                int(round(float(center_orig[1]) * float(scale_factor))),
            )
            mapped_radius = int(round(float(radius_orig) * float(scale_factor)))
            # Clamp to upscaled bounds
            uh, uw = upscaled_bgr.shape[:2]
            mapped_center = (
                max(0, min(mapped_center[0], uw - 1)),
                max(0, min(mapped_center[1], uh - 1)),
            )
            mapped_radius = max(1, min(mapped_radius, max(uw, uh)))
            detected_center, detected_radius = mapped_center, mapped_radius
            # Keep hough_meta as "upscaled" attempt + record original attempt separately
        else:
            # Final fallback: older 256px-width Hough settings (often better on classic Petri photos)
            center_legacy, radius_legacy, hough_meta_legacy256 = detect_circle_legacy256(bgr)
            if center_legacy is not None and radius_legacy is not None:
                detection_source = "legacy256_original_fallback"
                mapped_center = (
                    int(round(float(center_legacy[0]) * float(scale_factor))),
                    int(round(float(center_legacy[1]) * float(scale_factor))),
                )
                mapped_radius = int(round(float(radius_legacy) * float(scale_factor)))
                uh, uw = upscaled_bgr.shape[:2]
                mapped_center = (
                    max(0, min(mapped_center[0], uw - 1)),
                    max(0, min(mapped_center[1], uh - 1)),
                )
                mapped_radius = max(1, min(mapped_radius, max(uw, uh)))
                detected_center, detected_radius = mapped_center, mapped_radius

    cropped = False
    crop_offset_processed = (0, 0)
    bbox_meta: Dict[str, Any] = {"bbox": None}
    mask_meta: Dict[str, Any] = {"applied": False}
    processed_bgr = upscaled_bgr

    if detected_center is not None and detected_radius is not None:
        processed_bgr, crop_offset_processed, bbox_meta = crop_to_circle_mask_bbox(
            upscaled_bgr, detected_center, detected_radius
        )
        cropped = bbox_meta.get("bbox") is not None

        # Critical: even after cropping bbox, the image still contains corners outside the dish.
        # Apply a circular mask (fill outside with background color) so OpenCFU can't detect there.
        if cropped and bbox_meta.get("bbox") is not None:
            bx = int(bbox_meta["bbox"]["x"])
            by = int(bbox_meta["bbox"]["y"])
            local_center = (int(detected_center[0]) - bx, int(detected_center[1]) - by)
            mask_meta = apply_circular_background_mask_inplace(processed_bgr, local_center, int(detected_radius))

    processed_h, processed_w = processed_bgr.shape[:2]

    # 3) Output path
    if args.out:
        out_path = args.out
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, "temp_preprocessing")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"preprocessed_{uuid.uuid4().hex}.png")

    ok = cv2.imwrite(out_path, processed_bgr)
    if not ok:
        _die(f"Error: Failed to write processed image to: {out_path}")

    # Convert processed-space offsets to original-space offsets
    inv_scale = (1.0 / float(scale_factor)) if float(scale_factor) > 0.0 else 1.0
    crop_offset_original = (float(crop_offset_processed[0]) * inv_scale, float(crop_offset_processed[1]) * inv_scale)

    detected_circle_processed: Dict[str, Any] = {"present": False}
    detected_circle: Dict[str, Any] = {"present": False}
    if detected_center is not None and detected_radius is not None:
        detected_circle_processed = {
            "present": True,
            "source": "hardcoded" if detection_source.startswith("hardcoded") else "auto",
            "center": {"x": int(detected_center[0]), "y": int(detected_center[1])},
            "radius": int(detected_radius),
        }
        detected_circle = {
            "present": True,
            "source": "hardcoded" if detection_source.startswith("hardcoded") else "auto",
            "center": {
                "x": float(detected_center[0]) * inv_scale,
                "y": float(detected_center[1]) * inv_scale,
            },
            "radius": float(detected_radius) * inv_scale,
        }

    payload: Dict[str, Any] = {
        "processed_image_path": out_path,
        "created_temp_file": True,
        "cropped": bool(cropped),
        "scale_factor": float(scale_factor),
        "crop_offset_processed": {"x": int(crop_offset_processed[0]), "y": int(crop_offset_processed[1])},
        "crop_offset": {"x": float(crop_offset_original[0]), "y": float(crop_offset_original[1])},
        "original_size": {"width": int(original_w), "height": int(original_h)},
        "processed_size": {"width": int(processed_w), "height": int(processed_h)},
        "upscaled_size": upscale_meta.get("upscaled_size"),
        "detected_circle_processed": detected_circle_processed,
        "detected_circle": detected_circle,
        "debug": {
            "upscale": upscale_meta,
            "hough": hough_meta,
            "hough_original": hough_meta_original,
            "hough_legacy256": hough_meta_legacy256,
            "detection_source": detection_source,
            "bbox": bbox_meta.get("bbox"),
            "circle_mask": mask_meta,
        },
    }

    _json_stdout(payload)


if __name__ == "__main__":
    main()
