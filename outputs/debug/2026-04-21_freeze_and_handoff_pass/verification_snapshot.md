# Verification Snapshot

Date: 2026-04-21

## Verified source-of-truth artifacts

### 1. Official pattern single-model baseline

- Artifact:
  - `outputs/metrics/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/test_metrics.json`
- Verified metrics:
  - balanced accuracy: `0.8481921571352465`
  - macro F1: `0.7989648033126295`
- Result:
  - matches the expected freeze target

### 2. Best deployed pattern inference rule

- Artifact:
  - `outputs/metrics/pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo/test_metrics.json`
- Verified metrics:
  - balanced accuracy: `0.8563222384360595`
  - macro F1: `0.8114845938375351`
- Result:
  - matches the expected deployed freeze target

### 3. Latest TG rescue pass

- Artifact:
  - `outputs/debug/2026-04-21_post_unified_rescue_pass/summary.md`
  - `outputs/debug/2026-04-21_post_unified_rescue_pass/tg_experiment_summary.csv`
  - `outputs/debug/2026-04-21_post_unified_rescue_pass/selection_stability_report.csv`
- Verified conclusions:
  - structured TG remains more promising than flat TG
  - `TG-A1_serious_distill` is the best unrestricted serious rescue branch
  - `TG-A2_multiscale_distill_seed42` guardrail checkpoint is the best TG-safe research candidate
  - `type3` remains effectively unlearned
  - TG remains experimental

### 4. Latest severity structured / salvage passes

- Artifacts:
  - `outputs/debug/2026-04-20_severity_structured_pass/summary.md`
  - `outputs/debug/2026-04-21_post_unified_rescue_pass/summary.md`
  - `outputs/debug/2026-04-21_post_unified_rescue_pass/severity_experiment_summary.csv`
- Verified conclusions:
  - best known strict severity reference: BA `0.6109`, macro F1 `0.5542`
  - best salvage baseline: `hgb_fallback`, BA `0.3280`, macro F1 `0.3327`
  - no salvage model beat the prior best strict severity reference
  - severity should remain post-hoc / experimental

### 5. Unified 3-task pass

- Artifacts:
  - `outputs/debug/2026-04-21_unified_three_task_pass/summary.md`
  - `outputs/debug/2026-04-21_unified_three_task_pass/path_ranking.csv`
- Verified conclusions:
  - unified all-task training regressed pattern badly
  - the main unified artifact is not a recommended production path

### 6. Paper-style shadow benchmark

- Artifacts:
  - `outputs/debug/2026-04-21_paper_style_shadow_benchmark/summary.md`
  - `outputs/debug/2026-04-21_paper_style_shadow_benchmark/protocol_gap_audit.md`
  - `outputs/debug/2026-04-21_paper_style_shadow_benchmark/official_vs_shadow_comparison.csv`
- Verified conclusions:
  - the shadow pass is diagnostic only
  - it did not show that strict leakage-safe evaluation is the main reason for lower scores

## Discrepancy check

No discrepancy was found between the expected freeze targets and the saved official / deployed pattern metrics.
