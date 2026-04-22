# WORKING SEV-S3 Note

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

## What Changed

- Extended `build_geometry_table.py` with a few extra extent-focused features for SEV-S3:
  - `response_area_frac_t3_0`
  - `response_area_frac_t3_5`
  - `lesion_mass_top25_fraction`
  - `lesion_mass_top10_fraction`
  - `paracentral_minus_peripheral_fraction`
- Added `build_s2_ordinal_tables.py` to derive:
  - flat S2 noncentral extent table
  - `<=25` threshold table
  - `>=75` threshold table
- Added `train_s2_ordinal_tabular.py` for flat and threshold rules/HGB runs.
- Added `eval_s2_ordinal_severity.py` to compare:
  - flat S2 replacement
  - ordinal thresholded S2 replacement
  on top of frozen S0/S1 from SEV-S2.

## Best SEV-S3 Result

- `severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1`
- test balanced accuracy `0.4233`
- test macro F1 `0.4366`
- no-ulcer precision `1.0000`
- central-ulcer recall `0.6923`
- `ulcer_leq_25pct` recall / F1 `0.2000 / 0.1765`

## Comparison

- vs `hgb_fallback`: `+0.0953` BA, `+0.1038` macro F1
- vs best SEV-S1: `+0.0240` BA, `+0.0346` macro F1
- vs best SEV-S2: `+0.0020` BA, `+0.0018` macro F1
- vs strict reference: still `-0.1876` BA, `-0.1177` macro F1

## What Improved

- The ordinal S2 rescue improved the mildest noncentral class without breaking the useful SEV-S2 gates.
- `ulcer_leq_25pct` recall improved from `0.1333` to `0.2000`.
- `ulcer_leq_25pct` F1 improved from `0.1429` to `0.1765`.
- no-ulcer precision stayed `1.0000`.
- central-ulcer recall stayed `0.6923`.

## What Did Not Improve Enough

- The gain over best SEV-S2 is very small.
- Adjacent-class error worsened.
- Mild and medium noncentral cases are still often escalated upward.
- The noncentral extent split is still the bottleneck.

## Recommendation

- Continue only if the next pass stays very narrow:
  - S2 calibration
  - S2 probability coupling
  - conflict handling that is less biased toward `>=75`
- Do not reopen TG.
- Do not touch the frozen pattern line.
- Do not claim severity is solved.
