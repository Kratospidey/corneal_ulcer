# TG Punctate Audit

## Scope

- Experiment audited: `tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42`
- Pattern stays frozen and separate.
- This is diagnostic only. No new TG recipe search was performed.

## Exact Split Counts

| Split | no_ulcer | micro_punctate | macro_punctate | coalescent_macro_punctate | patch_gt_1mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 25 | 55 | 28 | 7 | 382 |
| val | 5 | 11 | 6 | 2 | 82 |
| test | 6 | 12 | 6 | 1 | 84 |

## Effective Counts Reaching Each TG Gate

| Split | T1 no_ulcer | T1 ulcer_present | T2 punctate_family | T2 patch_gt_1mm | T3 micro | T3 macro | T3 coalescent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 25 | 472 | 90 | 382 | 55 | 28 | 7 |
| val | 5 | 101 | 19 | 82 | 11 | 6 | 2 |
| test | 6 | 103 | 19 | 84 | 12 | 6 | 1 |

## TG-A3 Test Metrics

- balanced accuracy `0.3714`
- macro F1 `0.3528`
- punctate-family balanced accuracy `0.1389`
- punctate-family macro F1 `0.1961`

## Per-Class Recall / F1

- `no_ulcer`: `0.5000 / 0.4286`
- `micro_punctate`: `0.4167 / 0.4167`
- `macro_punctate`: `0.0000 / 0.0000`
- `coalescent_macro_punctate`: `0.0000 / 0.0000`
- `patch_gt_1mm`: `0.9405 / 0.9186`

## Full Test Confusion Matrix

- labels: `no_ulcer, micro_punctate, macro_punctate, coalescent_macro_punctate, patch_gt_1mm`
- matrix: `[[3, 3, 0, 0, 0], [2, 5, 1, 0, 4], [2, 0, 0, 0, 4], [0, 0, 0, 0, 1], [1, 4, 0, 0, 79]]`

## Punctate-Family-Only Confusion Matrix

- true rows: `micro_punctate, macro_punctate, coalescent_macro_punctate`
- predicted cols: `no_ulcer, micro_punctate, macro_punctate, coalescent_macro_punctate, patch_gt_1mm`
- matrix: `[[2, 5, 1, 0, 4], [2, 0, 0, 0, 4], [0, 0, 0, 0, 1]]`

## Direct Answers

### 1. How many `type3` examples exist in each split?

- train: `7`
- val: `2`
- test: `1`

### 2. Are `type3` failures mostly predicted as `type2`, `type1`, or swallowed by `type4`?

- `type3 -> type2`: `0`
- `type3 -> type1/no_ulcer`: `0`
- `type3 -> type4`: `1`

### 3. Does T2 gating starve T3?

- Yes. `type3_t2_starvation_share = 1.0000`.

### 4. Are there obvious label-noise / visual-ambiguity cases?

- Conflicting punctate duplicate groups: `0`
- The regenerated `type3` false-negative panel is visually confluent and patch-like, which supports some label-boundary ambiguity.
- That is not strong enough to make duplicate-driven label noise the primary explanation.

### 5. Is the punctate family visually separable enough to justify another learned rescue?

- Not from this line.
- `micro_punctate` has partial signal, but `macro_punctate` and `coalescent_macro_punctate` are effectively absent at recall level.
- Another tiny-loss rescue is not justified from these diagnostics.

## Audit Panels

- `outputs/debug/tg_punctate_audit/panels/type3_false_negatives.png`
- `outputs/debug/tg_punctate_audit/panels/punctate_true_positives.png`
- `outputs/debug/tg_punctate_audit/panels/type2_type3_confusions.png`
  - not emitted because no test predictions fell into a pure `type2 <-> type3` confusion bucket

## Verdict

- Main failure mode: `data scarcity + hierarchy starvation`, with a smaller amount of `label-boundary ambiguity` between severe punctate and patch-like appearances.
- Practical conclusion: do not spend more compute on tiny TG loss tweaks or more seeds from this foundation.