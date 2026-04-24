# Codex Working Truth

## Live Repo

- This repo is intentionally reduced to one task: `pattern_3class`.
- The active model line is ConvNeXtV2 Tiny only.
- Historical TG, severity, segmentation-assisted, external-warmstart, Swin, MaxViT, and proxy-geometry branches were removed.

## Current Winners

- Official single checkpoint:
  - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - balanced accuracy `0.8482`
  - macro F1 `0.7990`
- Best deployed inference rule:
  - `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
  - balanced accuracy `0.8563`
  - macro F1 `0.8115`

## Entry Points

- Train:
  - `PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_train.py --config configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__retrain_lineage_v1.yaml --device cuda`
- Evaluate checkpoint:
  - `PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_eval.py --config configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml --checkpoint models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt --split test --device cuda`
- Late-fusion inference:
  - `PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/run_late_fusion.py --config configs/inference_pattern_latefusion_v1.yaml --device cuda`

## Kept Surface

- `src/main_train.py`
- `src/main_eval.py`
- `src/run_late_fusion.py`
- `src/evaluation/paper_figures.py`
- the pattern ConvNeXt configs under `configs/`

## Guardrails

- Keep the task as `pattern_3class`.
- Keep the backbone family as `convnextv2_tiny` unless there is explicit new evidence.
- Do not confuse the official single checkpoint with the late-fusion deployment rule.
