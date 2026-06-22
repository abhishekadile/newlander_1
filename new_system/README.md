# new_system — YOLO26 Colony Detection

This directory contains a **YOLO26-based deep learning colony detection system**
built as a replacement candidate for the OpenCFU-based pipeline in the repository root.

> **Important:** Nothing in the repository root (`server.js`, `colonyDetector.js`,
> `preprocess.py`, `core_engine/`, etc.) is modified. The existing system remains
> fully runnable for benchmark comparison.

---

## Directory Layout

```
new_system/
├── data/
│   ├── raw/                      # Downloaded datasets (git-ignored)
│   │   ├── makrai2023/           # Makrai et al. 2023 (CC BY 4.0)
│   │   └── mcount/               # MCount dataset (CC0)
│   ├── processed/                # YOLO-format images + labels (git-ignored)
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   └── LICENSES.md               # Dataset license notices
├── scripts/
│   ├── download_datasets.py      # Phase 1 — download both datasets
│   ├── convert_to_yolo.py        # Phase 2 — convert annotations to YOLO format
│   ├── train_colony_model.py     # Phase 3 — YOLO26n training script
│   ├── run_colab_training.py     # Phase 4 — Colab CLI orchestration (Windows-compatible)
│   └── export_model.py           # Phase 5 — export to ONNX + OpenVINO
├── benchmark/
│   ├── run_benchmark.py          # Phase 6 — head-to-head benchmark runner
│   ├── current_system_adapter.py # Wrapper for existing OpenCFU system
│   ├── new_system_adapter.py     # Wrapper for YOLO26 + OpenVINO system
│   └── report/
│       └── benchmark_report.md   # Generated benchmark results
├── configs/
│   └── data.yaml                 # Ultralytics dataset config
├── weights/                      # Trained model weights (git-ignored)
└── runs/                         # Training logs + checkpoints (git-ignored)
```

---

## Prerequisites

```bash
pip install requests tqdm Pillow opencv-python numpy PyYAML ultralytics
# For benchmark only:
pip install openvino
# Colab CLI (already installed):
# uv tool install google-colab-cli  OR  pip install google-colab-cli
```

---

## Phase-by-Phase Reproduction

### Phase 0 — Already done (this scaffold)

### Phase 1 — Download Datasets

```bash
cd new_system/
python scripts/download_datasets.py
```

Downloads Makrai 2023 (Figshare) and MCount (Dryad) into `data/raw/`.
Writes `FORMAT_NOTES.md` per dataset and verifies image counts.

### Phase 2 — Convert to YOLO Format

```bash
python scripts/convert_to_yolo.py
```

Converts annotations to YOLO segmentation (preferred) or detection box format.
Writes `data/processed/`, `configs/data.yaml`, and `data/processed/CONVERSION_NOTES.md`.

To force detection-only (bounding boxes):
```bash
python scripts/convert_to_yolo.py --seg-fallback
```

### Phase 3 — Train (Locally, for testing)

```bash
python scripts/train_colony_model.py --device cpu --epochs 5 --batch 8
```

For full training, use Phase 4 (Colab T4 GPU).

### Phase 4 — Train on Colab T4 GPU

```bash
python scripts/run_colab_training.py --session colony_train --gpu T4
```

**First run**: A browser window opens for Google OAuth. Complete the login,
then press Enter in the terminal to continue.

The script:
1. Provisions a T4 session
2. Verifies GPU assignment (fails fast if CPU)
3. Mounts Google Drive for checkpoint persistence
4. Uploads dataset + configs
5. Installs Ultralytics on the VM
6. Runs training (≈5 hours for 150 epochs on T4)
7. Downloads checkpoints every 30 minutes (to `weights/last.pt`)
8. Downloads `best.pt` on completion
9. Saves session notebook to `runs/colab_session_log.ipynb`
10. **Always stops the session** (via try/finally — no lingering billed VMs)

To resume a disconnected session:
```bash
python scripts/run_colab_training.py --session colony_train --no-upload
```
The training script auto-resumes from `last.pt` if found on the VM.

### Phase 5 — Export Model

```bash
python scripts/export_model.py
```

Exports `weights/best.pt` to:
- `weights/best.onnx`
- `weights/best_openvino/` (directory with `.xml` + `.bin`)

### Phase 6 — Benchmark

First, start the existing server in another terminal:
```bash
# In d:/work/cell/ root:
node server.js
```

Then run:
```bash
python benchmark/run_benchmark.py
```

Results are written to `benchmark/report/benchmark_report.md`.

To run benchmark without the existing server (uses Node.js subprocess fallback):
```bash
python benchmark/run_benchmark.py
```

To benchmark only the new system:
```bash
python benchmark/run_benchmark.py --skip-current
```

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model | YOLO26n-seg (nano) | Smallest/fastest YOLO26 variant; validated before larger models |
| Train split | Makrai 80/10/10 | Standard ML split; MCount fully held out |
| MCount split | 100% test | Unbiased merged-colony evaluation; contamination would invalidate benchmark |
| Export format | OpenVINO | Matches target CPU deployment environment |
| Benchmark protocol | 5× timed + 1 warm-up | p95 latency more meaningful than best-case |
| Colab CLI | Python script (not bash) | Cross-platform; works on Windows without WSL2 |

---

## Licenses

See [`data/LICENSES.md`](data/LICENSES.md) for full attribution.

- **Makrai et al. 2023**: CC BY 4.0 — attribution required in publications
- **MCount**: CC0 — public domain
