# 2026-04-21 Official Freeze And Handoff

## Project Status Summary

This handoff freezes the current trustworthy project state after the TG rescue, severity salvage, unified three-task, and paper-style shadow passes.

The project has one strong mature result:
- `pattern_3class` on the official leakage-safe holdout

The rest of the active lines remain research lines:
- TG / type grading: promising but unstable
- severity: still secondary and unresolved
- unified 3-task training: research artifact only
- paper-style shadow benchmark: diagnostic only

## A. Current Official Freeze

### Official model

- Name: `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
- Task: `pattern_3class`
- Why it is official:
  - It is the best verified single-checkpoint result under the repo's leakage-aware holdout protocol.
  - Later experimental branches did not produce a trustworthy replacement.
- Verified test metrics:
  - balanced accuracy: `0.8481921571`
  - macro F1: `0.7989648033`
- Primary source artifacts:
  - `outputs/metrics/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/test_metrics.json`
  - `outputs/reports/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/test_summary.md`

### Best deployed inference rule

- Name: `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
- Why it is better for deployment:
  - It beats the official single checkpoint on the same task under the allowed inference-rule story.
  - It is the strongest currently frozen deployed pattern result.
- Why it is not the same as the official single checkpoint:
  - It is a late-fusion inference rule built from component prediction tables.
  - It is not a single trained checkpoint and should not replace the official single-model freeze.
- Verified test metrics:
  - balanced accuracy: `0.8563222384`
  - macro F1: `0.8114845938`
- Primary source artifacts:
  - `outputs/metrics/pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo/test_metrics.json`
  - `outputs/reports/pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo/test_summary.md`

## B. Task-by-Task Status

### Pattern

- Current official result:
  - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - test BA `0.8482`
  - test macro F1 `0.7990`
- Current best deployed result:
  - `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
  - test BA `0.8563`
  - test macro F1 `0.8115`
- Confidence level: high
- Recommended for real use: yes

### TG / type grading

- Best unrestricted serious rescue branch:
  - `TG-A1_serious_distill`
  - test BA `0.5102`
  - test macro F1 `0.4920`
  - pattern BA regressed relative to the official baseline
- Best TG-safe research candidate:
  - `TG-A2_multiscale_distill_seed42` guardrail checkpoint
  - TG BA `0.4970`
  - TG macro F1 `0.4500`
  - pattern BA `0.8388`
- Current conclusion:
  - structured TG is more promising than flat TG
  - `type3` remains effectively unlearned
  - seed sensitivity and validation/test mismatch remain real
  - no TG branch is stable enough to promote
- Confidence level: medium-low
- Recommended for real use: experimental only

### Severity

- Best known strict severity reference:
  - `severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_tempered_v1__holdout_v1__seed42` [unrestricted]
  - severity BA `0.6109`
  - severity macro F1 `0.5542`
  - not promoted because the branch was not accepted as the current official direction
- Best salvage baseline:
  - `hgb_fallback`
  - severity BA `0.3280`
  - severity macro F1 `0.3327`
- Current conclusion:
  - severity should remain post-hoc
  - no salvage model beat the prior best strict severity reference
  - learned end-to-end geometry heads failed
- Confidence level: low-medium
- Recommended for real use: post-hoc only / experimental

## C. What Failed

- Unified all-3-at-once training failed to preserve pattern.
- Learned severity geometry heads collapsed and lost to simpler alternatives.
- The paper-style shadow benchmark did not show that strict leakage-safe evaluation was the main cause of lower scores.
- TG remains dominated by punctate-family instability, especially `type3`.

## D. Recommended Next Step

- Keep the official pattern model unchanged.
- Keep the best deployed inference rule documented separately from the official checkpoint.
- Continue only with a narrow TG rescue pass if research time is available.
- Keep severity post-hoc.
- Do not reopen another broad unified pass first.

## E. Where The Evidence Lives

| Name | Task | Balanced accuracy | Macro F1 | Status | Note |
| --- | --- | ---: | ---: | --- | --- |
| `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42` | pattern | 0.8482 | 0.7990 | official | Current official single checkpoint |
| `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo` | pattern | 0.8563 | 0.8115 | deployed | Best deployed inference rule, not a single checkpoint |
| `TG-A1_serious_distill` | TG | 0.5102 | 0.4920 | experimental | Best unrestricted serious TG rescue branch |
| `TG-A2_multiscale_distill_seed42` guardrail checkpoint | TG | 0.4970 | 0.4500 | experimental | Best TG-safe research candidate |
| `severity_first_structured3head_tempered_v1` unrestricted | severity | 0.6109 | 0.5542 | experimental | Best known strict severity reference |
| `hgb_fallback` | severity | 0.3280 | 0.3327 | experimental | Best salvage baseline; severity remains post-hoc |
| `pattern3_tg5_severity5__...__unified_structured_tg_hybrid_severity_v1__...` | unified 3-task | pattern BA 0.7340, TG BA 0.4789, severity BA not competitive | n/a | diagnostic | Main unified research artifact; not recommended |
| `shadow_diag2024__vit_small_patch16_224__scenario1_pattern3__seed42` | paper-style shadow | 0.6857 | 0.7159 | diagnostic | Best shadow scenario result; not comparable to official results |

Primary evidence files:
- `outputs/debug/2026-04-21_post_unified_rescue_pass/summary.md`
- `outputs/debug/2026-04-21_post_unified_rescue_pass/tg_experiment_summary.csv`
- `outputs/debug/2026-04-21_post_unified_rescue_pass/severity_experiment_summary.csv`
- `outputs/debug/2026-04-21_post_unified_rescue_pass/selection_stability_report.csv`
- `outputs/debug/2026-04-20_severity_structured_pass/summary.md`
- `outputs/debug/2026-04-21_unified_three_task_pass/summary.md`
- `outputs/debug/2026-04-21_paper_style_shadow_benchmark/summary.md`
- `outputs/debug/2026-04-21_paper_style_shadow_benchmark/official_vs_shadow_comparison.csv`

## Practical Recommendation

If a collaborator needs one thing to use now:
- use the official pattern single checkpoint for official reporting
- use the late-fusion deployed rule if the goal is best current pattern inference
- treat TG and severity as experimental research outputs, not promoted deliverables
