# Dataset Licenses

## Makrai et al. 2023 — Bacterial Colony Detection Dataset

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
https://creativecommons.org/licenses/by/4.0/

**Citation (required):**
> Makrai, L. et al. Annotated dataset for deep-learning-based bacterial colony
> detection. Sci. Data 10, 497 (2023).
> Figshare https://doi.org/10.6084/m9.figshare.22022540.v3

**Usage in this project:**
The dataset is downloaded directly to the Colab VM during training (`remote_dataset_setup.py`).
It is never committed to this repository. The `new_system/data/` directory is git-ignored.

## MCount Dataset (Dryad)

**Status: INACCESSIBLE as of June 2026.**
The MCount dataset (merged/touching-colony images) is currently locked/inaccessible on Dryad.
It has not been downloaded or used in this pass. Merged-colony-specific validation cannot
be performed until access is restored or sufficient touching-colony images are collected
from production use.

**Action required:** Revisit once Dryad access is restored or enough real-world
touching/merged-colony images are collected from the IncuCountAPI deployment.
