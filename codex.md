# Codex Working Truth

## Active Scope

- This repo is intentionally reduced to the active `pattern_3class` line.
- Keep the codebase centered on the frozen ConvNeXtV2 Tiny pattern classifier and the deployed late-fusion rule.
- Archived TG, severity, segmentation-assisted, external-warmstart, Swin, MaxViT, and proxy-geometry branches were removed to keep the repo debuggable.

## Current Benchmarks

- Official single checkpoint:
  - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - balanced accuracy `0.8482`
  - macro F1 `0.7990`
- Best deployed inference rule:
  - `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
  - balanced accuracy `0.8563`
  - macro F1 `0.8115`

## What Remains In Scope

- `src/main_train.py`
- `src/main_eval.py`
- `src/run_late_fusion.py`
- `src/evaluation/paper_figures.py`
- the configs needed for:
  - raw ConvNeXtV2 Tiny training
  - cornea-crop ConvNeXtV2 Tiny training
  - late-fusion evaluation

## Guardrails

- Do not change the active task away from `pattern_3class`.
- Do not reintroduce deleted historical research branches unless there is a clear new need.
- Do not confuse the official single checkpoint with the deployed late-fusion rule.
