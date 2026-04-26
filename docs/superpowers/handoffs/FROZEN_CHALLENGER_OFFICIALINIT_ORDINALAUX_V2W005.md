# Frozen Challenger: officialinit_ordinalaux_v2w005

## Status

This is the best generated challenger checkpoint so far.

It does **not** replace the official frozen checkpoint yet.

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
| Accuracy | 0.8611 |
| Balanced accuracy | 0.8509 |
| Macro F1 | 0.8330 |
| Weighted F1 | 0.8614 |
| ECE | 0.0732 |

## Per-Class Recall Comparison

| Class | Official | Challenger | Delta |
| --- | ---: | ---: | ---: |
| point_like | 0.9630 | 0.9259 | -0.0371 |
| point_flaky_mixed | 0.6585 | 0.7805 | +0.1220 |
| flaky | 0.9231 | 0.8462 | -0.0769 |

## Why This Matters

The challenger is the first model-improvement result that beats the official checkpoint on test balanced accuracy while also improving macro F1 substantially.

It was produced from a known, current, reproducible code path:

- official checkpoint initialization
- ordinal auxiliary head
- ordinal auxiliary weight 0.05
- same `pattern_3class` task
- same `convnextv2_tiny` backbone family
- same frozen holdout split

## Caveat

The challenger's best validation balanced accuracy was still slightly below the official checkpoint's validation balanced accuracy.

Therefore this is frozen as a **challenger**, not promoted as the new official model.

## Exported Checkpoint

```text
models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_v2w005__holdout_v1__seed42/best.pt
```

## Config

```text
configs/model_improve/train_pattern3_officialinit_ordinalaux_v2_w005.yaml
```

## Required Next Experiments

1. Confirm the same recipe on additional seeds.
2. Try weight interpolation between the official checkpoint and this challenger.
3. Try smaller ordinal weights, especially 0.025 and 0.035.
4. Try gentle fine-tuning / partial freezing from the official checkpoint.
5. Only then consider dual-crop/context experiments.

## Promotion Rule

Do not promote unless a model beats the official checkpoint clearly and does not collapse per-class recall.

Current promotion bar:

- balanced accuracy > 0.8509 is better than this challenger
- balanced accuracy >= 0.8563 starts competing with the previous best deployed late-fusion rule
- macro F1 should remain >= 0.8330

## Export Metadata

The exported challenger directory also contains:

- `config.yaml`
- `test_metrics.json`
- `test_summary.md`
- `SHA256SUMS.txt`

The `best.pt` file is kept locally under the exported path above, but it remains ignored by Git under the repo's current model-artifact convention.
