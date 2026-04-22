# SEV-S3 No-Ulcer Precision Audit

- Final experiment: `severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1`
- S0 experiment: `severity5__posthoc__factorized_s0_plus_patternlogits_hgb_v1__holdout_v1`
- Audit verdict: `clean conservative behavior`

## Core Counts

- true no_ulcer support (test): `7`
- predicted no_ulcer count (test): `1`
- no_ulcer precision: `1.0000`
- no_ulcer recall: `0.1429`
- no_ulcer F1: `0.2500`

## Exact Test IDs

- predicted no_ulcer and correct: `['324']`
- true no_ulcer but missed: `['175', '280', '308', '36', '89', '94']`
- false no_ulcer predictions: `[]`

## S0 Behavior

- S0 confusion matrix: `[[1, 6], [0, 101]]`
- S0 predicted no_ulcer count (test): `1`
- final predicted no_ulcer count equals S0 no_ulcer count: `True`
- mean no_ulcer margin on true no_ulcer cases: `-0.4677`
- min no_ulcer margin on true no_ulcer cases: `-0.9998`
- max no_ulcer margin on missed true no_ulcer cases: `-0.0393`

## Leakage / Routing Checks

- duplicate image ids after merge: `0`
- raw image paths crossing splits: `0`
- cornea mask paths crossing splits: `0`
- suspicious numeric columns in geometry table: `[]`
- forbidden stage feature columns: `[]`
- pattern feature columns limited to logits/probs/confidence/interactions: `True`
- evaluator mentions factorized_route: `False`

## Interpretation

- The perfect no-ulcer precision is numerically fragile because the model predicts `no_ulcer` only once on the test split.
- That lone prediction is not produced by a combiner bug; it comes directly from the S0 gate and stays consistent through final routing.
- No obvious split leakage, duplicate leakage, target-derived numeric leakage, or factorized-route leakage was found in this audit.
- The behavior is conservative rather than suspicious: precision is perfect because predicted count is tiny, while recall is poor.