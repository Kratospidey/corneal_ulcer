# Frozen Challenger: officialinit_ordinalaux_w0035

## Status

This is the current best generated challenger checkpoint.

It does **not** replace the official frozen checkpoint yet.

Status framing:

- official = historical anchor
- `v2w005` = superseded generated challenger
- `w0035` = current best generated challenger

## Official Anchor

| Metric | Official checkpoint |
| --- | ---: |
| Accuracy | 0.8426 |
| Balanced accuracy | 0.8482 |
| Macro F1 | 0.7990 |
| Weighted F1 | 0.8439 |
| ECE | 0.0790 |

## Frozen Challenger

| Metric | Challenger |
| --- | ---: |
| Accuracy | 0.8796 |
| Balanced accuracy | 0.8671 |
| Macro F1 | 0.8546 |
| Weighted F1 | 0.8801 |
| ECE | 0.0728 |

## Per-Class Recall Comparison

| Class | Official | Challenger | Delta |
| --- | ---: | ---: | ---: |
| point_like | 0.9630 | 0.9259 | -0.0371 |
| point_flaky_mixed | 0.6585 | 0.8293 | +0.1708 |
| flaky | 0.9231 | 0.8462 | -0.0769 |

## Why This Matters

This is the first generated challenger that clears the previous generated line by a meaningful margin and also beats the official checkpoint on test balanced accuracy, macro F1, and accuracy.

It was produced from a known, current, reproducible code path:

- official checkpoint initialization
- ordinal auxiliary head
- ordinal auxiliary weight `0.035`
- same `pattern_3class` task
- same `convnextv2_tiny` backbone family
- same frozen holdout split

## Caveat

This result still needs confirmation on at least one additional seed before any official promotion.

The model improves the weak `point_flaky_mixed` boundary substantially, but it still trades away some `point_like` and `flaky` recall relative to the official checkpoint.

## Exported Checkpoint

```text
models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42/best.pt
```

## Config

```text
configs/model_improve/ordinal_weight_grid/train_pattern3_officialinit_ordinalaux_w0035.yaml
```

## Conservative Fallback

The best no-retrain single-checkpoint fallback from this pass is still:

- official/challenger checkpoint interpolation `alpha=0.20`
- balanced accuracy `0.8563`
- macro F1 `0.8115`

That candidate preserves official `point_like` and `flaky` recall better than `w0035`.

## Required Next Experiments

1. Confirm the same `w0035` recipe on additional seeds.
2. Run official versus `w0035` checkpoint interpolation.
3. Keep the `alpha=0.20` interpolation result archived as the conservative fallback.
4. Only then decide whether to promote `w0035`.

## Promotion Rule

Do not promote unless a candidate beats this challenger clearly or `w0035` survives confirmation without unacceptable recall collapse.

Current promotion bar:

- balanced accuracy >= `0.8671`
- macro F1 >= `0.8546`
- no catastrophic `flaky` recall collapse

## Export Metadata

The exported challenger directory also contains:

- `config.yaml`
- `test_metrics.json`
- `test_summary.md`
- `SHA256SUMS.txt`

The `best.pt` file is kept locally under the exported path above, but it remains ignored by Git under the repo's current model-artifact convention.
