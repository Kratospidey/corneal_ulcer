# Training Pipelines

This repository is frozen as a pattern-first project. The training stack remains leakage-aware, but only the pattern line should be treated as current and healthy.

## Supported Tasks

- Active:
  - `pattern_3class`
- Archived research artifacts:
  - `severity_5class`
  - `task_tg_5class`
  - `binary`

Do not treat TG or severity as current continuation targets.

## Split Policy

- Split files are project-generated under `data/interim/split_files/`
- Splits are built at raw-image ID level
- Duplicate candidates from `outputs/tables/duplicate_candidates.csv` are grouped before splitting
- Default holdout is `70/15/15`
- Augmentation is train-only

## Baseline Models

- `resnet18`
- `vgg16`
- `alexnet`

## ConvNeXtV2 Models

- `convnextv2_tiny`
- `convnextv2_base`

The official frozen benchmark line is:

- `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`

The best deployed inference rule is separate:

- `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`

## Main Commands

```bash
micromamba activate corneal-train

PYTHONPATH=src python src/main_eval.py \
  --config configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml \
  --checkpoint models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt \
  --split test \
  --device cuda

PYTHONPATH=src python src/run_ensemble_improvement_pass.py \
  --output-root outputs \
  --debug-root outputs/debug/2026-04-20_ensemble_improvement_pass
```

## Outputs

- metrics: `outputs/metrics/<experiment>/`
- reports: `outputs/reports/<experiment>/`
- confusion matrices: `outputs/confusion_matrices/<experiment>/`
- ROC curves: `outputs/roc_curves/<experiment>/`
- PR curves: `outputs/pr_curves/<experiment>/`
- predictions: `outputs/predictions/<experiment>/`
- explainability: `outputs/explainability/<experiment>/`
- checkpoints: `models/checkpoints/<experiment>/`

## Common Failure Points

- No official split files exist in the dataset; the project generates its own safe split files.
- No patient IDs exist; do not make patient-level generalization claims.
- Do not augment before splitting.
- Do not treat TG or severity as active training fronts here.
- Do not destabilize the official pattern benchmark.
