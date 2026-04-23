# MaxViT Ensemble Pattern Plan

## Scope

- Active task only: `pattern_3class`
- New single-model line:
  - `pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
- New ensemble lines:
  - `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_eq__holdout_v1__seed42`
  - `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_valtuned__holdout_v1__seed42`
- Explicitly excluded:
  - TG/type
  - severity/grade
  - segmentation-assisted or mask-derived paths
  - extra ensemble variants beyond equal-weight and one validation-tuned rule

## Live Pattern Pipeline Assumptions

- Canonical split:
  - `data/interim/split_files/pattern_3class_holdout.csv`
- Label column:
  - `task_pattern_3class`
- Label order:
  - `point_like`
  - `point_flaky_mixed`
  - `flaky`
- Shared crop and augmentation recipe:
  - `cornea_crop_scale_v1`
  - `pattern_augplus_v2`
- Shared sampling rule:
  - `weighted_sampler_tempered`
  - `sampler_temperature: 0.65`
- Shared checkpoint-selection metric:
  - validation `balanced_accuracy`
- Prediction export contract:
  - required columns:
    - `image_id`
    - `split`
    - `target_index`
    - `predicted_index`
    - `prob_point_like`
    - `prob_point_flaky_mixed`
    - `prob_flaky`
  - sidecar provenance:
    - `<split>_prediction_metadata.json`
    - contains `task_name`, fixed `class_names`, and fixed `probability_columns`

## Planned Runs

1. Train and evaluate `pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
2. Verify or regenerate compatible ConvNeXtV2 crop exports on `train`, `val`, and `test`
3. Run:
   - `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_eq__holdout_v1__seed42`
   - `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_valtuned__holdout_v1__seed42`
4. Compare honestly against:
   - frozen official single-model benchmark
   - frozen deployed late-fusion rule

## Leakage Checks

- preserve the canonical holdout split
- preserve the fixed label order across both models
- fail hard on export schema mismatch
- fail hard on provenance mismatch
- fail hard on sample coverage mismatch
- fail hard on `target_index` mismatch for the same `image_id`
- choose ensemble weights on validation only
- do not use TG/type, severity, segmentation, or mask-derived codepaths
- if MaxViT is weak alone, still run the required equal-weight and one validation-tuned ensemble, but do not expand the search surface
