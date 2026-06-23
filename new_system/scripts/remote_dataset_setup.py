"""
remote_dataset_setup.py — runs ON the Colab VM, NOT locally.

Executed via: colab exec -f scripts/remote_dataset_setup.py

Responsibilities:
  1. Download the Makrai 2023 bacterial colony dataset as a single bulk zip
     from Figshare (article 22022540 v3).
  2. Unzip and locate the COCO JSON annotation file.
  3. Read images.xls for per-image metadata (species_id, background flag).
  4. Stratified 80/10/10 train/val/test split by species × background.
  5. Convert COCO bounding boxes → YOLO label format.
  6. Mirror images + labels into /content/data_yolo/{train,val,test}/{images,labels}/.
  7. Write /content/configs/data.yaml.
  8. Log counts and sanity-check against known dataset totals.

Dataset: Makrai et al. 2023 (CC BY 4.0)
  https://doi.org/10.6084/m9.figshare.22022540.v3
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import openpyxl
import requests
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BULK_URL = "https://figshare.com/ndownloader/articles/22022540/versions/3"
FIGSHARE_API_URL = "https://api.figshare.com/v2/articles/22022540/versions/3"

RAW_DIR = Path("/content/data_raw")
ZIP_PATH = RAW_DIR / "makrai2023.zip"
UNZIP_DIR = RAW_DIR / "makrai2023"
YOLO_DIR = Path("/content/data_yolo")
CONFIG_DIR = Path("/content/configs")

# Expected dataset totals for sanity checking (10 % tolerance)
EXPECTED_IMAGES = 369
EXPECTED_INSTANCES = 56865
EXPECTED_SPECIES = 24
SANITY_TOLERANCE = 0.10  # 10 %

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[setup] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def check_tolerance(label: str, got: int, expected: int, tol: float = SANITY_TOLERANCE) -> None:
    lo = int(expected * (1 - tol))
    hi = int(expected * (1 + tol))
    if not (lo <= got <= hi):
        die(
            f"Sanity check FAILED for {label}: got {got}, expected {expected} "
            f"(tolerance ±{int(tol*100)}%, range [{lo}, {hi}]). "
            "Stop and inspect the downloaded archive before training on a bad parse."
        )
    log(f"Sanity check OK — {label}: {got} (expected ~{expected})")


# ---------------------------------------------------------------------------
# Phase 1: Download
# ---------------------------------------------------------------------------

def download_bulk_zip() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        log(f"Zip already present at {ZIP_PATH} — skipping download.")
        return

    log(f"Verifying bulk download URL via streaming GET: {BULK_URL}")
    # NOTE: Figshare's ndownloader returns 202 to HEAD requests while the zip is
    # being assembled. We must use a streaming GET and inspect the first bytes for
    # the PK zip magic (50 4B 03 04) rather than trusting HEAD Content-Type.
    try:
        with requests.get(BULK_URL, stream=True, allow_redirects=True, timeout=60) as probe:
            first_bytes = next(probe.iter_content(chunk_size=8), b"")
            content_length = int(probe.headers.get("Content-Length", 0))
            final_url = probe.url
            log(f"Probe GET: status={probe.status_code} final_url={final_url} "
                f"first_bytes={first_bytes[:4].hex()!r} Content-Length={content_length:,}")

        ZIP_MAGIC = b'\x50\x4b\x03\x04'
        min_expected_bytes = 100 * 1024 * 1024  # 100 MB
        is_zip = first_bytes[:4] == ZIP_MAGIC
        is_large = content_length >= min_expected_bytes

        if not is_zip:
            log("WARNING: Bulk URL response does not start with PK zip magic — "
                "response is not a zip file. Falling back to Figshare API enumeration.")
            _fallback_api_download()
            return
        if not is_large and content_length > 0:
            log(f"WARNING: Bulk zip appears small ({content_length / 1024**2:.1f} MB < 100 MB). "
                "Proceeding but verify after download.")
    except requests.RequestException as exc:
        log(f"WARNING: GET probe failed ({exc}). Falling back to Figshare API.")
        _fallback_api_download()
        return

    log(f"Downloading {content_length / 1024**2:.1f} MB to {ZIP_PATH} …")
    with requests.get(BULK_URL, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with ZIP_PATH.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {downloaded / 1024**2:6.1f} / {total / 1024**2:.1f} MB "
                              f"({pct:.1f}%)", end="", flush=True)
    print()  # newline after progress bar
    log(f"Download complete: {ZIP_PATH.stat().st_size / 1024**2:.1f} MB")


def _fallback_api_download() -> None:
    """Download individual files via the Figshare API — slow path, used only if bulk fails."""
    log("Fetching file list from Figshare API …")
    resp = requests.get(FIGSHARE_API_URL, timeout=30)
    resp.raise_for_status()
    article = resp.json()
    files = article.get("files", [])
    log(f"Found {len(files)} files via API.")
    UNZIP_DIR.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files, 1):
        name = f["name"]
        url = f["download_url"]
        dest = UNZIP_DIR / name
        if dest.exists():
            continue
        log(f"  [{i}/{len(files)}] Downloading {name} …")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    log("Fallback API download complete — skipping unzip step (files already in UNZIP_DIR).")


# ---------------------------------------------------------------------------
# Phase 2: Unzip
# ---------------------------------------------------------------------------

def unzip_archive() -> None:
    if UNZIP_DIR.exists() and any(UNZIP_DIR.iterdir()):
        log(f"Unzip dir already populated: {UNZIP_DIR} — skipping.")
        return
    UNZIP_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Unzipping {ZIP_PATH} → {UNZIP_DIR} …")
    result = subprocess.run(
        ["unzip", "-q", str(ZIP_PATH), "-d", str(UNZIP_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        die(f"unzip failed:\n{result.stderr}")
    log("Unzip complete.")


# ---------------------------------------------------------------------------
# Phase 3: Locate COCO JSON and images.xls
# ---------------------------------------------------------------------------

def find_coco_json() -> Path:
    candidates = list(UNZIP_DIR.rglob("*.json"))
    if not candidates:
        die("No JSON file found in the unzipped archive. Inspect the archive contents.")
    # Prefer the file with 'annotation' or 'coco' in its name
    ranked = sorted(candidates, key=lambda p: (
        0 if ("annotation" in p.name.lower() or "coco" in p.name.lower()) else 1,
        -p.stat().st_size,  # larger file is more likely the annotation file
    ))
    chosen = ranked[0]
    log(f"COCO annotation file: {chosen} ({chosen.stat().st_size / 1024**2:.1f} MB)")
    return chosen


def find_xls() -> Path:
    candidates = list(UNZIP_DIR.rglob("images.xls*"))
    if not candidates:
        die("images.xls not found in the unzipped archive.")
    log(f"Metadata XLS: {candidates[0]}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Phase 4: Read metadata from images.xls
# ---------------------------------------------------------------------------

def read_metadata(xls_path: Path) -> dict:
    """
    Returns a dict keyed by image filename:
      { "filename.jpg": {"species_id": "sp01", "bg_flag": "white"}, ... }

    Supports both .xls (via xlrd) and .xlsx (via openpyxl).
    The XLS columns are expected to include (case-insensitive match):
      filename, species_id (or species), background (or bg, bg_flag)
    """
    suffix = xls_path.suffix.lower()

    if suffix == ".xls":
        # Old binary XLS format — must use xlrd
        import xlrd
        wb = xlrd.open_workbook(str(xls_path))
        ws = wb.sheet_by_index(0)
        header = [str(ws.cell_value(0, c)).strip().lower() for c in range(ws.ncols)]
        log(f"XLS header (xlrd): {header}")

        def col(candidates: list) -> int:
            for cand in candidates:
                if cand in header:
                    return header.index(cand)
            die(f"Could not find any of {candidates} in XLS header {header}")

        idx_filename = col(["image_name", "filename", "file_name", "image", "name"])
        idx_species  = col(["label_name", "species_id", "species", "sp_id", "organism", "label"])
        idx_bg       = col(["background", "bg", "bg_flag", "background_flag", "color"])

        meta = {}
        for row_idx in range(1, ws.nrows):
            fname = str(ws.cell_value(row_idx, idx_filename)).strip()
            if not fname:
                continue
            species = str(ws.cell_value(row_idx, idx_species)).strip() or "unknown"
            bg      = str(ws.cell_value(row_idx, idx_bg)).strip().lower() or "unknown"
            meta[fname] = {"species_id": species, "bg_flag": bg}

    else:
        # Modern XLSX format — use openpyxl
        wb = openpyxl.load_workbook(str(xls_path), read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            die("images.xls is empty.")
        header = [str(c).strip().lower() if c else "" for c in rows[0]]
        log(f"XLSX header (openpyxl): {header}")

        def col(candidates: list) -> int:
            for cand in candidates:
                if cand in header:
                    return header.index(cand)
            die(f"Could not find any of {candidates} in XLSX header {header}")

        idx_filename = col(["image_name", "filename", "file_name", "image", "name"])
        idx_species  = col(["label_name", "species_id", "species", "sp_id", "organism", "label"])
        idx_bg       = col(["background", "bg", "bg_flag", "background_flag", "color"])

        meta = {}
        for row in rows[1:]:
            if not row or not row[idx_filename]:
                continue
            fname   = str(row[idx_filename]).strip()
            species = str(row[idx_species]).strip() if row[idx_species] else "unknown"
            bg      = str(row[idx_bg]).strip().lower() if row[idx_bg] else "unknown"
            meta[fname] = {"species_id": species, "bg_flag": bg}

    log(f"Metadata loaded: {len(meta)} images from XLS.")
    return meta


# ---------------------------------------------------------------------------
# Phase 5: Parse COCO JSON
# ---------------------------------------------------------------------------

def parse_coco(coco_path: Path) -> tuple:
    """Returns (images_by_id, annotations_by_image_id)."""
    log(f"Parsing COCO JSON ({coco_path.stat().st_size / 1024**2:.1f} MB) …")
    with coco_path.open() as fh:
        coco = json.load(fh)

    images_by_id = {img["id"]: img for img in coco["images"]}
    annotations_by_img: dict = {}
    for ann in coco["annotations"]:
        annotations_by_img.setdefault(ann["image_id"], []).append(ann)

    log(f"COCO: {len(images_by_id)} images, "
        f"{len(coco['annotations'])} annotations, "
        f"{len(coco.get('categories', []))} categories")
    return images_by_id, annotations_by_img


# ---------------------------------------------------------------------------
# Phase 6: Stratified split
# ---------------------------------------------------------------------------

def make_split(images_by_id: dict, meta: dict) -> dict:
    """
    Returns {"train": [img_id, ...], "val": [...], "test": [...]}.
    Stratification key: species_id × bg_flag.

    Strategy:
    - Main split (80/20): stratified by species×bg, small strata collapsed to "misc"
    - Val/test split (50/50 of the 20%): unstratified — too few samples per stratum
      for nested stratification to be reliable on a ~74-image temp set.
    """
    from collections import Counter
    ids = list(images_by_id.keys())

    def strat_key(img_id: int) -> str:
        fname = images_by_id[img_id]["file_name"]
        basename = Path(fname).name
        m = meta.get(basename) or meta.get(fname) or {}
        sp  = m.get("species_id", "unknown")
        bg  = m.get("bg_flag", "unknown")
        return f"{sp}__{bg}"

    labels = [strat_key(i) for i in ids]

    # Collapse strata with < 2 samples into "misc" so sklearn can stratify.
    # Keep collapsing until no singleton remains (handles cascading singletons).
    counts = Counter(labels)
    safe_labels = [lbl if counts[lbl] >= 2 else "misc" for lbl in labels]
    # Re-check: if "misc" itself is a singleton, collapse everything into "misc"
    counts2 = Counter(safe_labels)
    if counts2.get("misc", 0) < 2:
        safe_labels = ["misc" for _ in safe_labels]
    log(f"Stratification groups: {len(Counter(safe_labels))} (after collapsing singletons)")

    # 80 % train, 20 % temp — try stratified, fall back to random
    strat = safe_labels if len(Counter(safe_labels)) > 1 else None
    try:
        train_ids, temp_ids = train_test_split(
            ids, test_size=0.20, random_state=42, stratify=strat
        )
        log(f"Train/temp split: stratified={'yes' if strat else 'no (fallback)'}")
    except ValueError as exc:
        log(f"WARNING: Stratified split failed ({exc}). Using random split.")
        train_ids, temp_ids = train_test_split(
            ids, test_size=0.20, random_state=42
        )
    # 50 % of temp → val, 50 % → test — unstratified (temp set is ~74 images,
    # too small for reliable nested stratification)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=42
    )

    log(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    return {"train": train_ids, "val": val_ids, "test": test_ids}


# ---------------------------------------------------------------------------
# Phase 7: COCO → YOLO conversion + file layout
# ---------------------------------------------------------------------------

def coco_bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple:
    """COCO [x, y, w, h] (top-left origin) → YOLO normalised [cx, cy, w, h]."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def find_image_file(img_entry: dict, search_root: Path) -> Path | None:
    """Locate the actual image file regardless of subdirectory nesting."""
    fname = img_entry["file_name"]
    # Try exact relative path first
    candidate = search_root / fname
    if candidate.exists():
        return candidate
    # Try by basename only
    candidates = list(search_root.rglob(Path(fname).name))
    if candidates:
        return candidates[0]
    return None


def build_yolo_dataset(
    images_by_id: dict,
    annotations_by_img: dict,
    split_ids: dict,
    image_search_root: Path,
) -> dict:
    """
    Writes images + .txt labels into YOLO_DIR/{split}/{images,labels}/.
    Returns statistics dict.
    """
    stats = {s: {"images": 0, "instances": 0} for s in SPLITS}

    for split, img_ids in split_ids.items():
        img_out = YOLO_DIR / split / "images"
        lbl_out = YOLO_DIR / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        missing = 0
        for img_id in img_ids:
            img_entry = images_by_id[img_id]
            src = find_image_file(img_entry, image_search_root)
            if src is None:
                missing += 1
                continue

            dst_img = img_out / src.name
            dst_lbl = lbl_out / (src.stem + ".txt")
            shutil.copy2(src, dst_img)

            anns = annotations_by_img.get(img_id, [])
            iw = img_entry["width"]
            ih = img_entry["height"]
            with dst_lbl.open("w") as fh:
                for ann in anns:
                    cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], iw, ih)
                    fh.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            stats[split]["images"] += 1
            stats[split]["instances"] += len(anns)

        if missing:
            log(f"WARNING: {missing} image files not found for split '{split}' — skipped.")

    return stats


# ---------------------------------------------------------------------------
# Phase 8: Write data.yaml
# ---------------------------------------------------------------------------

def write_data_yaml() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = CONFIG_DIR / "data.yaml"
    content = textwrap.dedent(f"""\
        # Auto-generated by remote_dataset_setup.py on the Colab VM
        path: {YOLO_DIR}
        train: train/images
        val: val/images
        test: test/images

        nc: 1
        names:
          0: colony

        # Makrai et al. 2023 (CC BY 4.0)
        # https://doi.org/10.6084/m9.figshare.22022540.v3
    """)
    yaml_path.write_text(content)
    log(f"data.yaml written to {yaml_path}")


# ---------------------------------------------------------------------------
# Phase 9: Per-split / per-species / per-background logging
# ---------------------------------------------------------------------------

def log_statistics(images_by_id: dict, split_ids: dict, meta: dict, stats: dict) -> None:
    from collections import defaultdict

    log("=" * 60)
    log("DATASET STATISTICS")
    log("=" * 60)

    for split in SPLITS:
        s = stats[split]
        log(f"  {split:5s}: {s['images']:3d} images, {s['instances']:6d} instances")

    total_images    = sum(s["images"] for s in stats.values())
    total_instances = sum(s["instances"] for s in stats.values())
    log(f"  TOTAL: {total_images} images, {total_instances} instances")
    log("")

    # Per-species counts
    species_counts: dict = defaultdict(int)
    bg_counts: dict = defaultdict(int)
    for img_id in images_by_id:
        fname = Path(images_by_id[img_id]["file_name"]).name
        m = meta.get(fname, {})
        species_counts[m.get("species_id", "unknown")] += 1
        bg_counts[m.get("bg_flag", "unknown")] += 1

    log(f"Species breakdown ({len(species_counts)} distinct):")
    for sp, cnt in sorted(species_counts.items()):
        log(f"  {sp}: {cnt}")

    log(f"Background breakdown:")
    for bg, cnt in sorted(bg_counts.items()):
        log(f"  {bg}: {cnt}")

    log("=" * 60)

    # Sanity checks
    check_tolerance("total images",    total_images,    EXPECTED_IMAGES)
    check_tolerance("total instances", total_instances, EXPECTED_INSTANCES)
    if len(species_counts) < EXPECTED_SPECIES * (1 - SANITY_TOLERANCE):
        die(
            f"Sanity check FAILED: only {len(species_counts)} species found, "
            f"expected ~{EXPECTED_SPECIES}."
        )
    log(f"Sanity check OK — species: {len(species_counts)} (expected ~{EXPECTED_SPECIES})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=== Makrai 2023 Dataset Setup (remote, on Colab VM) ===")
    log(f"Working directory: {os.getcwd()}")

    # Step 1: Download
    download_bulk_zip()

    # Step 2: Unzip (skip if fallback path already populated UNZIP_DIR)
    if not (UNZIP_DIR.exists() and any(UNZIP_DIR.iterdir())):
        unzip_archive()
    else:
        log(f"Unzip dir already populated — skipping unzip.")

    # Step 3: Locate key files
    coco_path = find_coco_json()
    xls_path  = find_xls()

    # Step 4: Metadata
    meta = read_metadata(xls_path)

    # Step 5: Parse COCO
    images_by_id, annotations_by_img = parse_coco(coco_path)

    # Step 6: Stratified split
    split_ids = make_split(images_by_id, meta)

    # Step 7: Build YOLO dataset
    # Images live somewhere under UNZIP_DIR (search recursively)
    stats = build_yolo_dataset(images_by_id, annotations_by_img, split_ids, UNZIP_DIR)

    # Step 8: Write data.yaml
    write_data_yaml()

    # Step 9: Log + sanity check
    log_statistics(images_by_id, split_ids, meta, stats)

    log("")
    log("Dataset setup complete.")
    log(f"  YOLO data root : {YOLO_DIR}")
    log(f"  data.yaml      : {CONFIG_DIR / 'data.yaml'}")
    log("")
    log("NOTE: Pull data.yaml back to local machine with:")
    log(f"  colab download {CONFIG_DIR / 'data.yaml'} new_system/configs/data.yaml")


if __name__ == "__main__":
    main()
