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
PYTHONPATH=src python src/run_ensemble_improvement_pass.py \
  --output-root outputs \
  --debug-root outputs/debug/2026-04-20_ensemble_improvement_pass
```

Primary output locations:
- `outputs/debug/2026-04-20_ensemble_improvement_pass/`
- `outputs/metrics/pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo/`

## TG rescue branch

The latest TG rescue implementation and configs currently live on the research worktree branch, not this freeze branch. If you have that branch checked out, the entrypoint is:

```bash
PYTHONPATH=src python src/main_train_tg_rescue.py --config <tg_rescue_config.yaml> --device cuda
```

Frozen summary artifacts for the last rescue pass:
- `outputs/debug/2026-04-21_post_unified_rescue_pass/summary.md`
- `outputs/debug/2026-04-21_post_unified_rescue_pass/tg_experiment_summary.csv`
- `outputs/debug/2026-04-21_post_unified_rescue_pass/selection_stability_report.csv`

## Severity salvage branch

The latest severity salvage implementation and configs currently live on the research worktree branch, not this freeze branch. If you have that branch checked out, the entrypoint is:

```bash
PYTHONPATH=src python src/run_severity_salvage.py --config <severity_salvage_config.yaml>
```

Frozen summary artifacts for the last severity salvage pass:
- `outputs/debug/2026-04-21_post_unified_rescue_pass/summary.md`
- `outputs/debug/2026-04-21_post_unified_rescue_pass/severity_experiment_summary.csv`

## Where outputs land

- Official evals: `outputs/metrics/`, `outputs/reports/`, `outputs/predictions/`
- Debug / pass-level summaries: `outputs/debug/<date>_<pass_name>/`
- Exported checkpoints: `models/exported/<experiment_name>/`

## Things to be careful about

- Do not blur the official pattern checkpoint with the late-fusion deployed rule.
- Do not treat TG as official yet.
- Do not treat severity as solved.
- Do not use shadow benchmark results as if they were on the official leaderboard.
- Do not reopen a broad unified 3-task pass before TG is narrowed and severity stays post-hoc.
