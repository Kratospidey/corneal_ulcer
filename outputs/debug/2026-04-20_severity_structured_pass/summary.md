# Severity Structured Pass Summary

## Baselines Used

- Official pattern single-model baseline: `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - Val BA: 0.7253
  - Test BA: 0.8482
- Current best pattern inference rule: `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
  - Test BA: 0.8563
- Existing severity single-model baseline: `severity5__convnextv2_tiny__raw_rgb__holdout_v1__seed42`
  - Test severity BA: 0.4585
- Latest hierarchical severity reference: `pattern3_severity5__convnextv2_tiny__cornea_crop_scale_v1__multitask_baseline_v1__holdout_v1__seed42`
  - Test severity BA: 0.6021

## Main Findings

- The flat severity-first baseline failed. It was worse than the old severity baseline and never produced a validation-safe checkpoint.
- The structured 3-head formulation beat the flat 5-way baseline cleanly and was the first path to clear the validation pattern guardrail.
- The distance-aware extent loss did not help. Its safe checkpoint was weaker than the plain structured model, and its unrestricted checkpoint still missed the validation guardrail.
- The tempered sampler on the structured model produced the best raw severity checkpoint in the whole pass, but that checkpoint was rejected by the formal validation guardrail.

## Pass Results

- Rank 1: `severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_tempered_v1__holdout_v1__seed42` [unrestricted]
  Severity test: BA 0.6109, macro F1 0.5542, top confusion ulcer_geq_75pct->ulcer_leq_50pct (11).
  Pattern test: BA 0.8538, macro F1 0.8800, delta vs official single +0.0056.
  Decision tag: severity_improving_but_guardrail_rejected.

- Rank 2: `severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_ordinal_v1__holdout_v1__seed42` [unrestricted]
  Severity test: BA 0.5790, macro F1 0.4883, top confusion ulcer_geq_75pct->ulcer_leq_50pct (10).
  Pattern test: BA 0.7914, macro F1 0.7992, delta vs official single -0.0567.
  Decision tag: severity_improving_but_guardrail_rejected.

- Rank 3: `severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_v1__holdout_v1__seed42` [safe]
  Severity test: BA 0.5784, macro F1 0.4792, top confusion ulcer_geq_75pct->ulcer_leq_50pct (10).
  Pattern test: BA 0.8109, macro F1 0.8123, delta vs official single -0.0373.
  Decision tag: selected_by_val_guardrail_but_test_pattern_regressed.

- Rank 4: `severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_v1__holdout_v1__seed42` [unrestricted]
  Severity test: BA 0.5670, macro F1 0.5143, top confusion ulcer_geq_75pct->ulcer_leq_50pct (20).
  Pattern test: BA 0.8233, macro F1 0.8325, delta vs official single -0.0249.
  Decision tag: severity_improving_but_guardrail_rejected.

- Rank 5: `severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_tempered_v1__holdout_v1__seed42` [safe]
  Severity test: BA 0.5123, macro F1 0.3850, top confusion ulcer_leq_50pct->ulcer_geq_75pct (14).
  Pattern test: BA 0.8314, macro F1 0.8487, delta vs official single -0.0168.
  Decision tag: eligible_under_guardrail.

- Rank 6: `severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_ordinal_v1__holdout_v1__seed42` [safe]
  Severity test: BA 0.4876, macro F1 0.3550, top confusion ulcer_leq_50pct->ulcer_geq_75pct (12).
  Pattern test: BA 0.8366, macro F1 0.8309, delta vs official single -0.0116.
  Decision tag: eligible_under_guardrail.

- Rank 7: `severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_flat5_v1__holdout_v1__seed42` [unrestricted]
  Severity test: BA 0.4619, macro F1 0.3092, top confusion ulcer_leq_50pct->no_ulcer (14).
  Pattern test: BA 0.8097, macro F1 0.8304, delta vs official single -0.0385.
  Decision tag: selected_by_val_guardrail_but_test_pattern_regressed.

## Decision

- Best raw severity result: `severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_tempered_v1__holdout_v1__seed42` [unrestricted] with test severity BA 0.6109 and macro F1 0.5542.
- That raw severity result changes severity by +0.1524 BA vs the old severity baseline and +0.0088 vs the latest hierarchical severity reference.
- Best severity-safe result under the validation guardrail: `severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_v1__holdout_v1__seed42` [safe] with test severity BA 0.5784 and macro F1 0.4792.
- Best dual-output keep candidate that stayed within both the validation guardrail and the held-out pattern margin: `severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_tempered_v1__holdout_v1__seed42` [safe] with test severity BA 0.5123 and test pattern BA 0.8314.