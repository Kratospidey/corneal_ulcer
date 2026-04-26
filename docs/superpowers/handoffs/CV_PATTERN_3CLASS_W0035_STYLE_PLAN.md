# CV Pattern 3-Class w0035 Style Plan

## Why CV is added
The current fixed holdout test set is small, especially for the `flaky` class (only 13 test samples). A few changed predictions can significantly move the balanced accuracy. Adding a 10-fold cross-validation (CV) benchmark helps estimate the robustness of the training recipe across alternative stratified splits.

## What it measures
It estimates the robustness of the w0035-style training recipe (mean balanced accuracy, standard deviation, per-fold metrics, and per-class recall stability).

## What it does not measure
It does not replace the frozen holdout result. The frozen holdout split and frozen w0035 checkpoint remain the deployment benchmarks. It does not measure the exact performance of the w0035 checkpoint because each fold trains a fresh model.

## Non-leakage rule
The CV folds do not warm-start from the official or w0035 checkpoints. They are initialized from the normal timm/ImageNet/FCMAE pretrained ConvNeXtV2 Tiny source.

## Split generation method
Splits are generated using `StratifiedKFold` (10 folds, seed 42) via `src/data/make_cv_splits.py` since no sufficient patient/case grouping fields were available in the manifest. The splits preserve class distribution.

## Config locations
`configs/cv_pattern_3class/w0035_style/fold_00.yaml` to `fold_09.yaml`.

## Output locations
- Folds evaluation: `outputs/cv_pattern_3class/w0035_style/fold_XX/`
- Aggregate reports: `outputs/reports/cv_pattern_3class/w0035_style/`

## Current Status
CV splits and configs generated. Folds evaluated. Aggregate results available in `CV10_RESULTS.md`.
