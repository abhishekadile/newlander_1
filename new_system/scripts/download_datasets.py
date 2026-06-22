#!/usr/bin/env python3
"""
download_datasets.py
====================
Phase 1: Download both training datasets programmatically.

Datasets:
  1. Makrai et al. 2023 — Figshare DOI 10.6084/m9.figshare.22022540.v3
     → data/raw/makrai2023/
  2. MCount — Dryad DOI 10.5061/dryad.2280gb62f
     → data/raw/mcount/

After download, each directory gets a FORMAT_NOTES.md describing what was found.
Script exits non-zero if image counts deviate >20% from expected values.

Usage:
    python scripts/download_datasets.py [--skip-existing]

Requirements:
    pip install requests tqdm
"""

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

# ── Try to import optional deps gracefully ────────────────────────────────────
try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("ERROR: Missing dependencies. Run: pip install requests tqdm", file=sys.stderr)
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
NEW_SYSTEM_ROOT = SCRIPT_DIR.parent
DATA_RAW = NEW_SYSTEM_ROOT / "data" / "raw"

MAKRAI_DIR = DATA_RAW / "makrai2023"
MCOUNT_DIR = DATA_RAW / "mcount"

# ── Expected counts (fail-fast guard) ─────────────────────────────────────────
MAKRAI_EXPECTED_IMAGES = 369
MCOUNT_EXPECTED_IMAGES = 960
COUNT_TOLERANCE = 0.20  # ±20%

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ──────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "", skip_existing: bool = False) -> Path:
    """Stream-download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already exists.")
        return dest

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=desc or dest.name, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def extract_zip(zip_path: Path, dest_dir: Path):
    """Extract a zip archive into dest_dir."""
    print(f"  Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def count_images(directory: Path) -> int:
    """Recursively count image files in a directory."""
    return sum(
        1 for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )


def check_count(found: int, expected: int, name: str):
    """Fail if found count deviates more than tolerance from expected."""
    if expected == 0:
        return
    ratio = abs(found - expected) / expected
    if ratio > COUNT_TOLERANCE:
        print(
            f"\nERROR: {name} image count mismatch!\n"
            f"  Expected ~{expected}, found {found} "
            f"(deviation {ratio*100:.1f}% > {COUNT_TOLERANCE*100:.0f}% tolerance)\n"
            f"  Check the download and rerun. If the dataset was updated, "
            f"adjust EXPECTED counts in this script.",
            file=sys.stderr
        )
        sys.exit(1)
    print(f"  ✓ {name}: {found} images (expected ~{expected})")


# ──────────────────────────────────────────────────────────────────────────────
# Makrai 2023 — Figshare
# ──────────────────────────────────────────────────────────────────────────────

def download_makrai(skip_existing: bool):
    print("\n=== Makrai et al. 2023 (Figshare 22022540.v3) ===")
    MAKRAI_DIR.mkdir(parents=True, exist_ok=True)

    api_url = "https://api.figshare.com/v2/articles/22022540/files"
    print(f"  Listing files via Figshare API: {api_url}")
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    files = resp.json()
    print(f"  Found {len(files)} file(s) in the article:")
    for f in files:
        print(f"    • {f['name']}  ({f.get('size', '?')} bytes)  id={f['id']}")

    for fmeta in files:
        fname = fmeta["name"]
        furl = fmeta["download_url"]
        dest = MAKRAI_DIR / fname
        print(f"\n  Downloading: {fname}")
        download_file(furl, dest, desc=fname, skip_existing=skip_existing)

        # Auto-extract zips
        if fname.lower().endswith(".zip"):
            extract_dir = MAKRAI_DIR / Path(fname).stem
            extract_zip(dest, extract_dir)

    # Inventory
    n_images = count_images(MAKRAI_DIR)
    check_count(n_images, MAKRAI_EXPECTED_IMAGES, "Makrai 2023")

    # Inspect annotation format
    annotations = list(MAKRAI_DIR.rglob("*.xml")) + list(MAKRAI_DIR.rglob("*.json"))
    ann_summary = []
    if any(MAKRAI_DIR.rglob("*.xml")):
        ann_summary.append("Pascal VOC XML")
    if any(MAKRAI_DIR.rglob("*.json")):
        # Try to detect COCO format
        for jf in list(MAKRAI_DIR.rglob("*.json"))[:1]:
            try:
                with open(jf) as fp:
                    data = json.load(fp)
                if "annotations" in data and "categories" in data:
                    ann_summary.append("COCO JSON")
                    break
                ann_summary.append("JSON (unknown schema)")
            except Exception:
                ann_summary.append("JSON (unreadable)")
    if not ann_summary:
        ann_summary.append("No annotation files found — manual inspection required")

    write_format_notes(
        MAKRAI_DIR,
        dataset="Makrai et al. 2023",
        doi="10.6084/m9.figshare.22022540.v3",
        files=[f["name"] for f in files],
        n_images=n_images,
        annotation_format=", ".join(ann_summary),
        extra_notes=(
            "Colony annotations may be bounding boxes (Pascal VOC) or polygons (COCO).\n"
            "The convert_to_yolo.py script will attempt polygon extraction first.\n"
            "See CONVERSION_NOTES.md in data/processed/ for the final decision."
        )
    )
    print(f"\n  Makrai download complete. {n_images} images, FORMAT_NOTES.md written.")


# ──────────────────────────────────────────────────────────────────────────────
# MCount — Dryad
# ──────────────────────────────────────────────────────────────────────────────

def download_mcount(skip_existing: bool):
    print("\n=== MCount Dataset (Dryad 10.5061/dryad.2280gb62f) ===")
    MCOUNT_DIR.mkdir(parents=True, exist_ok=True)

    doi_encoded = "doi:10.5061%2Fdryad.2280gb62f"
    api_url = f"https://datadryad.org/api/v2/datasets/{doi_encoded}"
    print(f"  Fetching dataset metadata: {api_url}")

    resp = requests.get(api_url, timeout=30, headers={"Accept": "application/json"})
    resp.raise_for_status()
    meta = resp.json()

    # Get file listing from Dryad
    # Dryad v2: files listed under _links or via /versions endpoint
    version_id = None
    if "_embedded" in meta and "stash:versions" in meta["_embedded"]:
        versions = meta["_embedded"]["stash:versions"]
        if versions:
            version_id = versions[-1].get("id")

    if version_id is None:
        # Try the versions endpoint directly
        versions_url = f"https://datadryad.org/api/v2/datasets/{doi_encoded}/versions"
        vr = requests.get(versions_url, timeout=30, headers={"Accept": "application/json"})
        vr.raise_for_status()
        vdata = vr.json()
        versions_list = vdata.get("_embedded", {}).get("stash:versions", [])
        if versions_list:
            version_id = versions_list[-1].get("id")

    if version_id is None:
        print("  WARNING: Could not determine Dryad version ID. Trying direct download URL.")
        # Dryad provides a direct download endpoint for the whole dataset
        direct_url = f"https://datadryad.org/api/v2/datasets/{doi_encoded}/download"
        dest = MCOUNT_DIR / "mcount_dataset.zip"
        print(f"  Downloading full dataset bundle: {dest.name}")
        download_file(direct_url, dest, desc="mcount_dataset.zip", skip_existing=skip_existing)
        if dest.exists():
            extract_zip(dest, MCOUNT_DIR)
    else:
        files_url = f"https://datadryad.org/api/v2/versions/{version_id}/files"
        print(f"  Listing files for version {version_id}: {files_url}")
        fr = requests.get(files_url, timeout=30, headers={"Accept": "application/json"})
        fr.raise_for_status()
        fdata = fr.json()
        file_list = fdata.get("_embedded", {}).get("stash:files", [])
        print(f"  Found {len(file_list)} file(s):")
        for fi in file_list:
            print(f"    • {fi.get('path', fi.get('id'))}  ({fi.get('size', '?')} bytes)")

        for fi in file_list:
            fname = fi.get("path", f"mcount_{fi.get('id', 'unknown')}")
            furl = fi.get("_links", {}).get("stash:download", {}).get("href", "")
            if not furl:
                print(f"  WARNING: no download URL for {fname}, skipping.")
                continue
            dest = MCOUNT_DIR / Path(fname).name
            print(f"\n  Downloading: {Path(fname).name}")
            download_file(furl, dest, desc=Path(fname).name, skip_existing=skip_existing)
            if dest.suffix.lower() == ".zip":
                extract_zip(dest, MCOUNT_DIR / dest.stem)

    # Inventory
    n_images = count_images(MCOUNT_DIR)
    check_count(n_images, MCOUNT_EXPECTED_IMAGES, "MCount")

    # Inspect annotation format
    ann_formats = []
    if any(MCOUNT_DIR.rglob("*.csv")):
        ann_formats.append("CSV (likely contour/centroid data)")
    if any(MCOUNT_DIR.rglob("*.json")):
        ann_formats.append("JSON")
    if any(MCOUNT_DIR.rglob("*.mat")):
        ann_formats.append("MATLAB .mat (likely blob/segment masks)")
    if any(MCOUNT_DIR.rglob("*.png")) and any(MCOUNT_DIR.rglob("*mask*")):
        ann_formats.append("PNG mask files")
    if not ann_formats:
        ann_formats.append("No annotation files detected — inspect manually")

    write_format_notes(
        MCOUNT_DIR,
        dataset="MCount",
        doi="10.5061/dryad.2280gb62f",
        files=[],  # Populated dynamically above
        n_images=n_images,
        annotation_format=", ".join(ann_formats),
        extra_notes=(
            "MCount is used as a HELD-OUT merged-colony evaluation set only.\n"
            "It is NOT mixed into training/val splits.\n"
            "Filenames will be prefixed with 'mcount_' during conversion to YOLO format.\n"
            "Annotations may be contour coordinates (CSV) or instance mask images."
        )
    )
    print(f"\n  MCount download complete. {n_images} images, FORMAT_NOTES.md written.")


# ──────────────────────────────────────────────────────────────────────────────
# FORMAT_NOTES.md writer
# ──────────────────────────────────────────────────────────────────────────────

def write_format_notes(
    directory: Path,
    dataset: str,
    doi: str,
    files: list,
    n_images: int,
    annotation_format: str,
    extra_notes: str = "",
):
    notes = (
        f"# Format Notes — {dataset}\n\n"
        f"**DOI:** {doi}\n\n"
        f"## Files Downloaded\n\n"
    )
    if files:
        notes += "\n".join(f"- `{f}`" for f in files) + "\n\n"
    else:
        notes += "_(see directory listing)_\n\n"

    notes += (
        f"## Image Count\n\n"
        f"{n_images} image files found (extensions: {', '.join(IMAGE_EXTS)})\n\n"
        f"## Annotation Format\n\n"
        f"{annotation_format}\n\n"
    )
    if extra_notes:
        notes += f"## Notes\n\n{extra_notes}\n"

    out = directory / "FORMAT_NOTES.md"
    out.write_text(notes, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download Makrai 2023 and MCount datasets.")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip files that already exist (resume interrupted download)."
    )
    parser.add_argument(
        "--dataset", choices=["makrai", "mcount", "both"], default="both",
        help="Which dataset to download. Default: both."
    )
    args = parser.parse_args()

    if args.dataset in ("makrai", "both"):
        download_makrai(skip_existing=args.skip_existing)
    if args.dataset in ("mcount", "both"):
        download_mcount(skip_existing=args.skip_existing)

    print("\n✓ All datasets downloaded and verified.")


if __name__ == "__main__":
    main()
