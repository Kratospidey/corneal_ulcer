# Prepush Verification

Date: 2026-04-21

## Verification checks

1. Docs exist
   - `docs/superpowers/handoffs/2026-04-21-official-freeze-and-handoff.md`
   - `docs/superpowers/handoffs/OFFICIAL_RESULTS_SUMMARY.md`
   - `docs/superpowers/handoffs/EXPERIMENTAL_BRANCH_STATUS.md`
   - `docs/superpowers/handoffs/RUNBOOK.md`
   - `outputs/debug/2026-04-21_freeze_and_handoff_pass/verification_snapshot.md`

2. Official model and deployed inference rule are clearly distinguished
   - verified in `docs/superpowers/handoffs/2026-04-21-official-freeze-and-handoff.md`
   - verified in `docs/superpowers/handoffs/OFFICIAL_RESULTS_SUMMARY.md`

3. Referenced artifact files exist on the publication branch
   - `outputs/metrics/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/test_metrics.json`
   - `outputs/metrics/pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo/test_metrics.json`
   - `outputs/debug/2026-04-21_post_unified_rescue_pass/summary.md`
   - `outputs/debug/2026-04-20_severity_structured_pass/summary.md`
   - `outputs/debug/2026-04-21_unified_three_task_pass/summary.md`
   - `outputs/debug/2026-04-21_paper_style_shadow_benchmark/summary.md`

4. Metrics match saved artifacts
   - official pattern checkpoint: BA `0.8481921571`, macro F1 `0.7989648033`
   - deployed late-fusion rule: BA `0.8563222384`, macro F1 `0.8114845938`
   - TG and severity branch summaries match the copied pass summaries referenced in `verification_snapshot.md`

5. No unrelated files are included in the publication worktree
   - current publication branch changes are limited to:
     - `docs/superpowers/handoffs/*`
     - copied small evidence artifacts under `outputs/debug/*`
     - copied official/deployed metric snapshots under `outputs/metrics/*` and `outputs/reports/*`

## Publish decision

Safe to stage and publish only the freeze / handoff package on branch `freeze/official-handoff-2026-04-21`.
