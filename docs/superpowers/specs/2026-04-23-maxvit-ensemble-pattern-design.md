# MaxViT Ensemble Pattern Design

## Goal

Add one clean image-only complementary ensemble experiment beside the frozen ConvNeXtV2 crop benchmark:

- train `MaxViT-Tiny` for `pattern_3class`
- export compatible `train` / `val` / `test` predictions
- ensemble it with the frozen ConvNeXtV2 crop model in probability space
- choose the ensemble weight on validation only
- evaluate once on test against the frozen official benchmark and the frozen deployed late-fusion rule

This design explicitly excludes TG/type, severity, segmentation, mask-derived features, pseudo-labeling, and broad model-zoo expansion.

## Frozen Truth To Preserve

- Active task family: `pattern_3class`
- Official single-model benchmark:
  - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - frozen test balanced accuracy `0.8482`
  - frozen test macro F1 `0.7990`
- Best deployed inference rule:
  - `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
  - frozen test balanced accuracy `0.8563`
  - frozen test macro F1 `0.8115`

The official checkpoint and the deployed late-fusion rule remain distinct in code, docs, and reporting.

## Live Pattern Pipeline Assumptions

The new MaxViT line must reuse the existing pattern-only discipline already used by the live crop benchmark.

- Canonical split file:
  - `data/interim/split_files/pattern_3class_holdout.csv`
- Label column:
  - `task_pattern_3class`
- Label order:
  - `point_like`
  - `point_flaky_mixed`
  - `flaky`
- Crop / preprocessing mode:
  - `cornea_crop_scale_v1`
- Training augmentation profile:
  - `pattern_augplus_v2`
- Train sampler:
  - `weighted_sampler_tempered`
- Sampler temperature:
  - `0.65`
- Checkpoint selection metric:
  - validation `balanced_accuracy`
- Prediction export format:
  - `outputs/predictions/<experiment>/<split>_predictions.csv`
  - probability columns emitted in the fixed class order above
- Ensemble execution path:
  - existing probability-space `src/run_late_fusion.py`

## Chosen Approach

Use the current pattern stack with the smallest possible extension:

1. add `MaxViT-Tiny` support to the existing `timm` model factory path
2. add one MaxViT training config that mirrors the frozen crop recipe
3. reuse `src/main_train.py` and `src/main_eval.py` for execution and export
4. reuse the existing probability-space fusion utility
5. add an explicit prediction-alignment guard inside the fusion path so leakage and ordering issues fail hard instead of being silently ignored

This keeps the experiment directly comparable to the deployed late-fusion and Swin ensemble artifacts and avoids introducing simultaneous changes such as logit-space fusion.

## New Artifacts

### Single Model

- `pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`

### Ensembles

- `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_eq__holdout_v1__seed42`
- `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_valtuned__holdout_v1__seed42`

The naming stays honest:

- `avgprob` because fusion is in probability space
- `valtuned` only for the validation-selected weighted rule

## Planned Code Changes

### Model Support

- Update `src/model_factory.py`
  - accept `maxvit_*` model names in the existing `timm` branch
  - add MaxViT support in the backbone-freezing helper so the family integrates cleanly with the current conventions

### Configs

- Create `configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml`
  - base config: same raw pattern base used by the existing crop lines
  - model id: `maxvit_tiny_tf_224.in1k`
  - preprocessing, augmentation, sampler, and promotion references aligned to the current crop benchmark
- Create `configs/inference_pattern_convnext_maxvit_avgprob_eq.yaml`
  - fixed weights `[0.5, 0.5]`
- Create `configs/inference_pattern_convnext_maxvit_avgprob_valtuned.yaml`
  - validation-only grid search
  - scalar weight grid step `0.05`

### Ensemble Guard

- Update `src/run_late_fusion.py`
  - validate class order equality across components
  - validate no duplicate `image_id` within each component export
  - validate identical sample coverage rather than silently intersecting mismatched sets
  - validate matching `target_index` per `image_id`
  - fail with a clear error if any alignment check fails

### Working Docs

- Create `docs/superpowers/handoffs/MAXVIT_ENSEMBLE_PATTERN_PLAN.md`
- Create `docs/superpowers/handoffs/MAXVIT_ENSEMBLE_PATTERN_RESULTS.md`
- Update `README_training.md` and `codex.md` only if the run path is real and reproducible in this repo snapshot

## Execution Plan

1. verify the live pattern assumptions above against the current code and artifacts
2. implement MaxViT model/config support
3. run MaxViT training on the canonical pattern split
4. export MaxViT predictions for `train`, `val`, and `test`
5. verify or regenerate compatible ConvNeXtV2 crop predictions for `train`, `val`, and `test`
6. run the equal-weight probability ensemble
7. run the validation-tuned probability ensemble
8. freeze the selected validation-tuned rule and evaluate it once on test
9. write the results handoff and minimal reproducibility doc updates

## Fairness And Leakage Rules

The implementation and final report must explicitly verify:

- canonical split preserved
- label mapping preserved across both models
- no sample-order mismatch between paired prediction exports
- no duplicate sample IDs inside export tables
- no ensemble weight selection on test
- no preprocessing fit on full-dataset labels
- no train/val/test prediction mixing
- no TG/type codepath used
- no severity codepath used
- no segmentation or mask-derived feature path used

If any mismatch is found, fix it before reporting results.

## Reporting Rules

The final results table must report, for each of the following:

1. frozen official single-model benchmark
2. frozen deployed late-fusion rule
3. new MaxViT-Tiny single model
4. new equal-weight ensemble
5. new validation-tuned ensemble

Required metrics:

- validation balanced accuracy
- validation macro F1
- test balanced accuracy
- test macro F1

Required comparisons:

- delta vs frozen official single-model benchmark
- delta vs frozen deployed late-fusion rule

If MaxViT is weak, report that plainly.
If the ensemble gain is small, report that plainly.
If neither ensemble beats the deployed late-fusion rule, report that plainly and do not promote it.

## Failure Policy

- If MaxViT training cannot complete, record the exact command, exact error, and exact environment path instead of presenting placeholder benchmark claims.
- If the `timm` family string differs at runtime, record the exact resolved model id honestly in the results doc.
- If the current local ConvNeXt artifact under the frozen experiment path differs from frozen truth, report the local rerun separately and do not overwrite the frozen benchmark values.

## Acceptance Gate

This work is only complete when all of the following are true:

- the MaxViT single-model line exists and is runnable
- MaxViT was actually executed
- compatible prediction exports exist for MaxViT and ConvNeXtV2
- equal-weight ensemble outputs were actually computed
- validation-tuned ensemble outputs were actually computed
- the tuned weight was selected on validation only
- the final report contains real metrics and exact experiment names
- frozen truth remains honest and clearly separated from the new ensemble artifacts
