# WORKING SEV-S2 Note

## Branch State

- Worktree: `/tmp/corneal_ulcer_sev_s1_audit`
- Branch: `exp/sev-s2-factorized-posthoc`
- Base working branch: `exp/sev-s1-geometry-audit`
- Pattern frozen: yes
- TG paused: yes

## Environment

- `Python 3.11.15`
- `torch 2.10.0+cu128`
- `torch.cuda.is_available() == True`
- `sklearn 1.8.0`
- `xgboost` unavailable

## What Changed

- Extended `build_geometry_table.py` to produce `geom_table_v2.csv` with richer response and interaction features.
- Added `build_factorized_tables.py` to derive S0/S1/S2 routed tables.
- Added `train_factorized_tabular.py` for stage-wise rules and HGB baselines.
- Added `eval_factorized_severity.py` for deterministic routed full-severity evaluation.
- Fixed a real SEV-S2 evaluation bug:
  - stage predictions cannot be read only from ground-truth stage subsets
  - S1 and S2 must be callable on misrouted samples too
  - HGB artifacts are now saved as `model.joblib`
  - the evaluator now runs stage inference across the full split

## Best Result

- `severity5__posthoc__factorized_geom_plus_patternlogits_hgb_v1__holdout_v1`
- test balanced accuracy `0.4213`
- test macro F1 `0.4347`
- no-ulcer precision `1.0000`
- central-ulcer recall `0.6923`

## Comparison

- vs `hgb_fallback`: `+0.0933` BA, `+0.1020` macro F1
- vs best SEV-S1: `+0.0220` BA, `+0.0328` macro F1
- vs strict reference: still `-0.1896` BA, `-0.1195` macro F1

## Remaining Weakness

- S2 is still the bottleneck.
- `ulcer_leq_25pct` recall is only `0.1333`.
- Many `ulcer_leq_25pct` and `ulcer_leq_50pct` cases still get escalated to `ulcer_geq_75pct`.
- The no-ulcer gain came mostly from becoming extremely conservative:
  - precision `1.0`
  - recall `0.1429`

## Recommendation

- Continue severity only if the next pass is still narrow and explicitly S2-focused.
- Do not reopen TG.
- Do not touch the frozen pattern line.
- Next plausible move:
  - improve noncentral extent scoring and calibration
  - not another broad model zoo
  - not a shared-backbone severity retrain
