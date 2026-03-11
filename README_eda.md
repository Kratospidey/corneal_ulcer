# EDA Workflow

This repository’s Stage 2 workflow builds a paper-grounded, leakage-aware exploratory data analysis pipeline for the SUSTech-SYSU corneal ulcer dataset. It is designed to validate dataset structure, supported tasks, preprocessing choices, and split risks before Stage 3 classification code is generated or trained.

## Environment

Use the existing micromamba environment, not `venv`.

```bash
micromamba activate corneal-train
python -m pip install -r requirements-eda.txt
python -m ipykernel install --user --name corneal-train --display-name "Python (corneal-train)"
```

The code is written for Python 3.11 and expects the `corneal-train` environment described in the project notes. PyTorch GPU support should already be fixed in that environment; the EDA runner still detects device availability at runtime and falls back cleanly to CPU where needed.

The Stage 2 embedding pass is now aligned to `ConvNeXtV2` by default, using `convnextv2_tiny` for latent-space EDA and one Diagnostics-inspired comparison representation (`masked_highlight_proxy`).

## Main Entry Point

Run the full EDA pipeline with:

```bash
micromamba activate corneal-train
python src/run_eda.py \
  --data-root data/raw/sustech_sysu \
  --output-root outputs \
  --device auto
```

Useful flags:

```bash
python src/run_eda.py --skip-embeddings
python src/run_eda.py --skip-duplicates
python src/run_eda.py --skip-preprocessing
python src/run_eda.py --skip-mask-analysis
python src/run_eda.py --device cpu
```

## Notebook

The narrative notebook lives at `notebooks/01_paper_grounded_dataset_eda.ipynb`.

It is intentionally thin and reuses the Python modules under `src/` instead of duplicating the implementation. The notebook is suitable for thesis-style review after the CLI runner has produced artifacts in `outputs/` and `data/interim/`.

## What The Runner Produces

Interim data:
- `data/interim/manifests/manifest.csv`
- `data/interim/manifests/manifest.parquet` when Parquet dependencies are available
- `data/interim/cleaned_metadata/audit_summary.csv`
- `data/interim/cleaned_metadata/label_summary.csv`
- `data/interim/cleaned_metadata/folder_tree.csv`

Output tables:
- `outputs/tables/manifest.csv`
- `outputs/tables/dataset_audit.csv`
- `outputs/tables/label_distributions.csv`
- `outputs/tables/image_stats.csv` when Pillow is available
- `outputs/tables/mask_stats.csv` when mask analysis runs
- `outputs/tables/duplicate_candidates.csv` when duplicate checks run
- `outputs/tables/embedding_summary.csv` when embeddings run
- `outputs/tables/embedding_projection_summary.csv` when embedding projections run
- `outputs/tables/small_data_risks.csv`

Output reports:
- `outputs/reports/dataset_audit_report.md`
- `outputs/reports/eda_summary.md`
- `outputs/reports/leakage_report.md`
- `outputs/reports/split_recommendations.md`
- `outputs/reports/model_readiness_summary.md`
- `outputs/reports/image_stats_summary.md`
- `outputs/reports/preprocessing_comparison_summary.md`
- `outputs/reports/embedding_analysis_summary.md`
- `outputs/reports/runtime_summary.json`

Figures:
- label distributions
- resolution, aspect-ratio, file-size, brightness, contrast, blur, entropy, and green-dominance histograms
- random sample grids and class-wise montages
- preprocessing montages and comparison grid
- mask overlay examples when masks are available
- ConvNeXtV2 embedding scatter plots and outlier panels

## Grounding Rules

The implementation follows the Stage 1 notes in `notes/`:
- dataset files are treated as source of truth
- paper claims are guidance, not project results
- supported tasks are only those confirmed from the actual dataset
- augmentation is never treated as pre-split
- mask availability is treated as leakage-sensitive because it is label-correlated

## Dependency Fallbacks

The main path expects the full CV stack from `requirements-eda.txt`, but the runner degrades gracefully:
- if `pyarrow` is missing, Parquet export is skipped
- if `Pillow` is missing, image-dependent stages are skipped
- if `torch` or `numpy` are missing, embeddings are skipped
- if `matplotlib` is missing, figure creation is skipped

This makes the CLI importable and partially runnable even before the environment is fully provisioned.
