# Training Pipelines

This repository is frozen as a pattern-first project. The training stack remains leakage-aware, but only the pattern line should be treated as current and healthy.

## Supported Tasks

- Active:
  - `pattern_3class`
- Archived research artifacts:
  - `severity_5class`
  - `task_tg_5class`
  - `binary`

Active training, evaluation, inference, and explainability entrypoints now reject archived tasks by default.

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
- `maxvit_tiny_tf_224.in1k`
- `swin_tiny_patch4_window7_224`

The official frozen benchmark line is:

- `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`

The best deployed inference rule is separate:

- `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`

Executed Swin ensemble follow-up:

- single-model Swin-T:
  - `pattern3__swin_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
- two-model image-only ensembles:
  - `pattern3__convnextv2_tiny_plus_swin_tiny__avgprob_eq__holdout_v1__seed42`
  - `pattern3__convnextv2_tiny_plus_swin_tiny__avgprob_valtuned__holdout_v1__seed42`
- current verdict:
  - clean and reproducible, but not promoted because neither ensemble beat the frozen pattern baselines

Executed MaxViT ensemble follow-up:

- single-model MaxViT-Tiny:
  - `pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
- two-model image-only ensembles:
  - `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_eq__holdout_v1__seed42`
  - `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_valtuned__holdout_v1__seed42`
- current verdict:
  - reproducible and cleaner than the Swin follow-up, but still not promoted because neither ensemble beat the frozen official benchmark or the deployed late-fusion rule

## Main Commands

```bash
micromamba activate corneal-train

PYTHONPATH=src python src/main_eval.py \
  --config configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml \
  --checkpoint models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt \
  --split test \
  --device cuda

PYTHONPATH=src python src/run_late_fusion.py \
  --config configs/inference_pattern_latefusion_v1.yaml \
  --device cuda

PYTHONPATH=src python src/main_train.py \
  --config configs/train_swin_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml \
  --device cuda

PYTHONPATH=src python src/main_train.py \
  --config configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml \
  --device cuda

PYTHONPATH=src python src/run_late_fusion.py \
  --config configs/inference_pattern_convnext_swin_avgprob_eq.yaml \
  --device cuda

PYTHONPATH=src python src/run_late_fusion.py \
  --config configs/inference_pattern_convnext_swin_avgprob_valtuned.yaml \
  --device cuda

PYTHONPATH=src python src/run_late_fusion.py \
  --config configs/inference_pattern_convnext_maxvit_avgprob_eq.yaml \
  --device cuda

PYTHONPATH=src python src/run_late_fusion.py \
  --config configs/inference_pattern_convnext_maxvit_avgprob_valtuned.yaml \
  --device cuda

PYTHONPATH=src python src/main_train_segmentation.py \
  --config configs/train_segmentation_ulcer_cornea.yaml \
  --device cuda

PYTHONPATH=src python src/main_infer_segmentation.py \
  --config configs/infer_segmentation_ulcer_cornea.yaml \
  --checkpoint models/checkpoints/seg2d__ulcer_cornea__supervised__holdout_v1__seed42/best.pt \
  --device cuda

PYTHONPATH=src python src/main_extract_predicted_mask_stats.py \
  --config configs/extract_predicted_mask_stats.yaml

PYTHONPATH=src python src/experimental/pattern/train_predmask_classifier.py \
  --config configs/train_pattern_embedding_plus_predmask_stats.yaml
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
- Do not promote the exploratory `0.8743` frozen-feature artifact as a like-for-like replacement for the official image-only benchmark.
- Do not use ground-truth mask-derived stats in the downstream pattern classifier if the intended inference path only has predicted masks.

## Segmentation-Assisted Experimental Track

- Segmenter experiment:
  - `seg2d__ulcer_cornea__supervised__holdout_v1__seed42`
- Downstream predicted-mask experiments:
  - `pattern3__predmaskstats_only__logreg__holdout_v1__seed42`
  - `pattern3__convnextv2_tiny_embeddings_only__logreg_pca256__holdout_v1__seed42`
  - `pattern3__convnextv2_tiny_plus_predmaskstats__hgb__holdout_v1__seed42`
  - `pattern3__convnextv2_tiny_logits_plus_predmaskstats__logreg__holdout_v1__seed42`

Promotion rule:

- valid and potentially promotable:
  - predicted-mask-derived features for train, val, and test from one frozen segmenter path
- non-promotable:
  - any downstream table that uses human ulcer-mask stats directly
