# Final Severity Status

## Results

- SEV-S1:
  - `severity5__posthoc__geom_plus_patternlogits_hgb_v1__holdout_v1`
  - balanced accuracy `0.3993`
  - macro F1 `0.4020`
- SEV-S2:
  - `severity5__posthoc__factorized_geom_plus_patternlogits_hgb_v1__holdout_v1`
  - balanced accuracy `0.4213`
  - macro F1 `0.4347`
- SEV-S3:
  - `severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1`
  - balanced accuracy `0.4233`
  - macro F1 `0.4366`
  - no-ulcer precision `1.0000`
  - central-ulcer recall `0.6923`
  - `ulcer_leq_25pct` recall / F1 `0.2000 / 0.1765`
- SEV-S4:
  - no-ulcer audit: clean conservative behavior, not leakage
  - no SEV-S4 variant beat SEV-S3 cleanly

## Final Winner

- Final severity fallback:
  - `severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1`
  - balanced accuracy `0.4233`
  - macro F1 `0.4366`

## Why The `1.0000` No-Ulcer Precision Was Not Leakage

- true `no_ulcer` test support was `7`
- predicted `no_ulcer` count was `1`
- the single predicted `no_ulcer` case was correct
- S0 itself predicted `no_ulcer` exactly once on test
- final `no_ulcer` predictions matched the S0 gate exactly
- no duplicate leakage, split leakage, target-derived geometry leakage, route leakage, or combiner bug was found

## Why Severity Continuation Is Stopped

- SEV-S3 was only a narrow improvement over SEV-S2.
- SEV-S4 did not produce a clean continuation:
  - `s0cal` increased BA but broke no-ulcer precision and hurt useful gates
  - `s2cal` helped the mildest class somewhat but lost too much overall performance
  - `probcombine` hurt macro F1 and no-ulcer precision
- The next plausible severity step would require a broader redesign than this branch allows.

Severity remains unresolved and is abandoned as an active continuation line on this foundation.

