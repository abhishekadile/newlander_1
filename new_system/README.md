# new_system — YOLO26-based Colony Detection

Replacement candidate for the OpenCFU-based pipeline in `IncuCountAPI`.
All training and dataset work runs on a Google Colab T4 GPU via the official
[google-colab-cli](https://github.com/googlecolab/google-colab-cli).
**No existing files outside `new_system/` are modified.**

---

## Directory Layout

```
new_system/
├── data/
│   └── LICENSES.md              # CC BY 4.0 attribution (required)
├── scripts/
│   ├── remote_dataset_setup.py  # [REMOTE] download + convert dataset on Colab VM
│   ├── train_colony_model.py    # [REMOTE] train YOLO26n on Colab VM
│   ├── export_model.py          # [REMOTE] export best.pt → ONNX + OpenVINO
│   ├── run_colab_training.sh    # [LOCAL]  full orchestration script
│   ├── monitor_training.py      # [LOCAL]  live epoch/loss/mAP tail + ETA
│   └── sync_checkpoints.py      # [LOCAL]  periodic best.pt/last.pt sync
├── benchmark/
│   ├── current_system_adapter.py
│   ├── new_system_adapter.py
│   ├── run_benchmark.py
│   └── report/
│       └── benchmark_report.md
├── configs/
│   └── data.yaml                # local reference; authoritative copy lives on VM
├── weights/                     # synced from Colab (git-ignored)
└── runs/                        # logs pulled from Colab (git-ignored)
```

---

## Prerequisites

```bash
# Install the Google Colab CLI (released June 2026)
pip install google-colab-cli   # or follow https://github.com/googlecolab/google-colab-cli

# Local Python deps for monitoring/benchmark
pip install requests openpyxl numpy
```

For local benchmark inference, also install:
```bash
pip install ultralytics openvino
```

---

## Phase 0 — Verify scaffold
All files in `new_system/` except `weights/` and `runs/` are committed.

## Phase 1 — Dataset (on Colab VM)
`remote_dataset_setup.py` is uploaded and executed on the VM.
It downloads the Makrai 2023 bulk archive from Figshare, converts to YOLO format,
performs a stratified 80/10/10 split by species × background, and writes `data.yaml`.

## Phase 2 — Train (on Colab VM)
`train_colony_model.py` trains YOLO26n (detection, bbox-only) with auto-resume support.
Training params are tuned for a small-scene dataset (~369 images, 56k+ instances).

## Phase 3 — Orchestrate + Monitor (local)
```bash
# Terminal 1: run the full orchestration (interactive — requires OAuth login)
bash new_system/scripts/run_colab_training.sh

# Terminal 2: live progress monitoring
python new_system/scripts/monitor_training.py --tier free

# Terminal 3: periodic checkpoint sync (every 10 min)
python new_system/scripts/sync_checkpoints.py
```

The orchestration script will pause and prompt you to complete Google OAuth login
in your browser on first use of the Colab CLI.

## Phase 4 — Export (on Colab VM)
```bash
colab exec -f new_system/scripts/export_model.py
colab download runs/yolo26n_colony/weights/best_openvino_model new_system/weights/
colab download runs/yolo26n_colony/weights/best.onnx new_system/weights/
```

## Phase 5 — Benchmark
```bash
# Requires: existing Express server running on localhost:3000 (or it falls back to direct call)
# Requires: new_system/weights/best.pt (or .xml/.onnx) synced from Colab
python new_system/benchmark/run_benchmark.py \
    --test-dir new_system/data/processed/test \
    --sample-dir images \
    --output new_system/benchmark/report/benchmark_report.md
```

---

## Known Limitations

1. **Merged/touching colonies (MCount):** The MCount dataset (Dryad) is inaccessible
   as of June 2026. Merged-colony performance has NOT been evaluated. Revisit when
   access is restored or when sufficient touching-colony images are collected from
   production.

2. **Glare / variable lighting:** Makrai 2023 provides only white-background and
   black-background plate images. Real-world glare and lighting diversity beyond
   this binary variation are unconfirmed in training data.

See `benchmark/report/benchmark_report.md` (after running Phase 5) for the detailed
tested-vs-unverified breakdown.

---

## Dataset License

Makrai et al. 2023 is CC BY 4.0. See `data/LICENSES.md` for the required attribution.
The dataset is never committed to this repository — `new_system/data/` is git-ignored.
