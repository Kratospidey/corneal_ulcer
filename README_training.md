# Training Pipelines

This repository’s training stack is leakage-aware and starts from the finished Stage 2 manifest, duplicate checks, split recommendations, and Stage 3 benchmark reports rather than rebuilding dataset understanding from scratch.

## Supported Tasks

- `pattern_3class`: primary first task
- `severity_5class`: supported in code, run only after the 3-class benchmark is stable
- `binary`: supported in code

Deferred from the default benchmark:
- `task_tg_5class`
- mask-assisted classification

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

The official Stage 3 reference baseline is `pattern3__alexnet__raw_rgb__holdout_v1__seed42`.
Phase 4 compares ConvNeXtV2 against that exact holdout benchmark instead of inventing a new split.

## Main Commands

```bash
micromamba activate corneal-train

python src/main_train.py --config configs/train_resnet18_raw.yaml
python src/main_train.py --config configs/train_vgg16_raw.yaml
python src/main_train.py --config configs/train_alexnet_raw.yaml

python src/main_train.py --config configs/train_resnet18_paper.yaml
python src/main_train.py --config configs/train_vgg16_paper.yaml
python src/main_train.py --config configs/train_alexnet_paper.yaml

python src/main_train.py --config configs/train_convnextv2_raw.yaml
python src/main_train.py --config configs/train_convnextv2_paper.yaml
python src/main_train.py --config configs/train_convnextv2_strong.yaml

python src/main_eval.py --config configs/train_resnet18_raw.yaml --checkpoint models/checkpoints/pattern3__resnet18__raw_rgb__holdout_v1__seed42/best.pt
python src/explainability/explain.py --config configs/explainability.yaml --train-config configs/train_resnet18_raw.yaml --checkpoint models/checkpoints/pattern3__resnet18__raw_rgb__holdout_v1__seed42/best.pt
python src/inference/predict.py --config configs/inference.yaml --checkpoint models/checkpoints/pattern3__resnet18__raw_rgb__holdout_v1__seed42/best.pt --image-path data/raw/sustech_sysu/rawImages/1.jpg
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
- Do not treat Stage 2 embedding findings as model performance claims.
- Use balanced accuracy, macro F1, and per-class recall as the main metrics, especially for severity.
- For ConvNeXtV2, keep raw RGB as the primary path and use `masked_highlight_proxy` only as a controlled comparison on the same split.
