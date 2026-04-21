# Protocol Gap Audit

## Purpose
This pass is a **paper-style shadow benchmark**, not an official benchmark refresh.

Its purpose is to estimate how much of the score gap may come from:
- the repo's stricter leakage-safe evaluation protocol
- the Diagnostics 2024-style preprocessing and looser evaluation setup
- the backbone itself

## Official repo protocol
Current official repo behavior is stricter than the paper-style benchmark:
- duplicate-aware splitting via `outputs/tables/duplicate_candidates.csv`
- group integrity enforced across train/val/test
- task-specific official holdout split files under `data/interim/split_files/`
- strong current image path built around `cornea_crop_scale_v1`
- current strongest official single-model line centered on `convnextv2_tiny`
- official promotion rule for the mature task is balanced accuracy first, then macro F1
- recent rescue work showed meaningful validation/test mismatch even under the strict protocol

## Diagnostics 2024 paper protocol
From the Diagnostics 2024 paper on the same public SUSTech-SYSU dataset:
- preprocessing removes the blue channel
- images are converted to grayscale
- Gaussian blur is applied
- Otsu thresholding is used
- masked, information-containing regions are retained and highlighted
- augmentation includes rotation, scaling, padding, and flipping
- the classifier is a Vision Transformer
- validation uses 10-fold cross-validation
- the paper reports accuracy, precision, recall, F1, and AUC for three scenarios:
  - pattern / type scenario
  - grade / TG scenario
  - severity scenario

## What this shadow pass matches faithfully
- same public SUSTech-SYSU manifest and label spaces
- same three task scenarios:
  - `pattern_3class`
  - `task_tg_5class`
  - `severity_5class`
- shadow preprocessing path isolated as `diagnostics2024_proxy_preproc_v1`
- blue-channel removal, grayscale conversion, Gaussian blur, Otsu thresholding, masking
- paper-style augmentation family isolated as `diagnostics2024_shadow_v1`
- a ViT-family shadow baseline
- a matched `convnextv2_tiny` comparison under the exact same shadow preprocessing and split protocol

## What this shadow pass approximates
- The paper uses 10-fold cross-validation. This pass does **not** run full 10-fold CV across all scenarios and backbones.
- Instead, `diagnostics2024_shadow_v1` uses a single stratified random 70/15/15 split with:
  - no duplicate-group leakage control
  - no group integrity enforcement
  - the same split reused across backbones for each scenario
- The paper's final masking presentation is green-highlighted. The proxy variant keeps only the masked signal in the green channel, which is close but still an implementation proxy rather than an exact figure-level recreation.
- The paper does not expose enough implementation detail to recover its exact ViT size, optimizer settings, or stopping rule. This pass therefore uses a practical ViT-family baseline (`vit_small_patch16_224`) rather than claiming exact architectural reproduction.

## What remains uncertain
- whether the paper augmented before or within cross-validation in a way that could further loosen evaluation
- whether the reported metrics are fold averages over all 10 folds or derived from another aggregation method
- exact ViT configuration, regularization, and optimization hyperparameters
- whether any hidden leakage exists in the paper protocol beyond the absence of duplicate-aware grouping

## Bottom line
This shadow protocol is deliberately looser than the official repo benchmark and is **not directly comparable** to the official leakage-safe holdout.

It is appropriate only for diagnosing:
- protocol effect
- preprocessing effect
- backbone effect under a looser paper-style setup

It is **not** appropriate for replacing the official benchmark or changing the official recommendation files.
