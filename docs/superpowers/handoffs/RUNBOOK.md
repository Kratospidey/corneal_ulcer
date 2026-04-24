# Runbook

## Environment

- micromamba environment: `corneal-train`
- activate:

```bash
micromamba activate corneal-train
```

- GPU note:
  - prefer `--device cuda` for evaluation and training
  - recent rescue and shadow passes were run on GPU

## Official pattern evaluation

Re-evaluate the current official single checkpoint:

```bash
PYTHONPATH=src python src/main_eval.py \
  --config configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml \
  --checkpoint models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt \
  --split test \
  --device cuda
```

Primary output locations:
- `outputs/metrics/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/`
- `outputs/reports/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/`

## Best deployed inference rule evaluation

Refresh the late-fusion deployment sweep:

```bash
PYTHONPATH=src python src/run_late_fusion.py \
  --config configs/inference_pattern_latefusion_v1.yaml \
  --device cuda
```

Primary output locations:
- `outputs/metrics/pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo/`
- `outputs/reports/pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo/`

## Exploratory follow-up artifact

- `pattern3__cornea_crop_scale_v1__convnextv2_tiny_plus_vit_small__stats__nearest_centroid__holdout_v1__seed42`
- treat it as exploratory only
- do not use it as the official image-only benchmark replacement
- its uplift depends on mask-stat features rather than a plain fine-tuned checkpoint

## Archived lines

- TG / type is archived on this foundation.
- Severity / grade is archived on this foundation.
- Do not use this repo freeze to continue minor TG or severity tweaks.

## Where outputs land

- Official evals: `outputs/metrics/`, `outputs/reports/`, `outputs/predictions/`
- Debug / pass-level summaries: `outputs/debug/<date>_<pass_name>/`
- Exported checkpoints: `models/exported/<experiment_name>/`

## Things to be careful about

- Do not blur the official pattern checkpoint with the late-fusion deployed rule.
- Do not reopen TG here.
- Do not reopen severity here.
- Do not use shadow benchmark results as if they were on the official leaderboard.
- Do not destabilize the official pattern line.
