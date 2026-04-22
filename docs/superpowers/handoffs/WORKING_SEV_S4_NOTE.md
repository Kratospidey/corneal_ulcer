# WORKING SEV-S4 Note

## Branch State

- Worktree: `/tmp/corneal_ulcer_sev_s1_audit`
- Branch: `exp/sev-s3-s2-ordinal-rescue`
- Pattern frozen: yes
- TG paused: yes
- Severity post-hoc only: yes

## Environment

- `Python 3.11.15`
- `torch 2.10.0+cu128`
- `torch.cuda.is_available() == True`
- `sklearn 1.8.0`
- `xgboost` unavailable

## Phase A Result

- SEV-S3 no-ulcer precision `1.0000` is real but fragile.
- Test support is only `7`.
- Predicted no-ulcer count is only `1`.
- The single predicted no-ulcer case is correct and comes directly from S0.
- No obvious leakage, split bug, route bug, or evaluator bug was found.

## Phase B Result

New variants tested:

- `severity5__posthoc__factorized_geom_plus_patternlogits_s0cal_hgb_v1__holdout_v1`
- `severity5__posthoc__factorized_geom_plus_patternlogits_s2cal_hgb_v1__holdout_v1`
- `severity5__posthoc__factorized_geom_plus_patternlogits_probcombine_hgb_v1__holdout_v1`

Best carried-forward baseline remains:

- `severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1`
- balanced accuracy `0.4233`
- macro F1 `0.4366`

What happened:

- `s0cal` improved BA to `0.4948` but destroyed no-ulcer precision and harmed central / mild behavior.
- `s2cal` improved `ulcer_leq_25pct` recall / F1 and adjacent error, but lost too much BA and macro F1.
- `probcombine` improved BA and adjacent error, but badly damaged no-ulcer precision and macro F1.

## Recommendation

- Stop severity continuation.
- SEV-S3 is the last promotable post-hoc fallback in this line.
- Do not spend more time on tiny calibration / combiner tweaks.
- Any next severity step would need a broader reformulation than this branch allows.
