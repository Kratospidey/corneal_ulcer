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

## Best Generated Challenger

- Status framing:
  - official = historical anchor
  - `v2w005` = superseded generated challenger
  - `w0035` = current best generated challenger

- `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42`
- balanced accuracy `0.8671`
- macro F1 `0.8546`
- accuracy `0.8796`
- weighted F1 `0.8801`
- ECE `0.0728`

This is the best generated/reproducible challenger so far, produced by warm-starting from the official checkpoint and training with a smaller ordinal auxiliary loss (`ordinal_aux_weight=0.035`).

It does **not** replace the official checkpoint yet because the result still needs seed confirmation.

Exported challenger path:

- `models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42/best.pt`

Previous frozen challenger:

- `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_v2w005__holdout_v1__seed42`
- balanced accuracy `0.8509`
- macro F1 `0.8330`
- superseded by `w0035`

Best no-retrain checkpoint interpolation candidate:

- official/challenger weight interpolation `alpha=0.20`
- balanced accuracy `0.8563`
- macro F1 `0.8115`
- preserves official `point_like` and `flaky` recall

## Entry Points

- Train:
  - `PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_train.py --config configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__retrain_lineage_v1.yaml --device cuda`
- Evaluate checkpoint:
  - `PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_eval.py --config configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml --checkpoint models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt --split test --device cuda`
- Late-fusion inference:
  - `PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/run_late_fusion.py --config configs/inference_pattern_latefusion_v1.yaml --device cuda`
- Evaluate frozen challenger:
  - `PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_eval.py --config configs/model_improve/ordinal_weight_grid/train_pattern3_officialinit_ordinalaux_w0035.yaml --checkpoint models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42/best.pt --split test --device cuda`

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
