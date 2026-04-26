# CV Results (w0035-style recipe)

The fixed holdout result remains the frozen deployment benchmark. The new cross-validation benchmark estimates the robustness of the w0035-style training recipe across alternative stratified splits.

## Fixed Holdout Anchors
- official: BA 0.8482, macro F1 0.7990
- w0035: BA 0.8671, macro F1 0.8546

The CV result is not directly identical to the fixed holdout because each fold trains a fresh model from normal pretrained initialization and evaluates on a different split. It estimates recipe robustness, not the exact w0035 checkpoint performance.

## Summary Statistics
| Metric | Mean | Std | Min | Max | Median |
|---|---|---|---|---|---|
| Val BA | 0.7188 | 0.0289 | 0.6667 | 0.7756 | 0.7195 |
| Test BA | 0.7109 | 0.0795 | 0.5370 | 0.8597 | 0.7230 |
| Test Macro F1 | 0.6555 | 0.0749 | 0.4996 | 0.7794 | 0.6541 |
| Test Weighted F1 | 0.7082 | 0.0705 | 0.5614 | 0.8121 | 0.7112 |
| Accuracy | 0.6964 | 0.0746 | 0.5352 | 0.8028 | 0.6972 |
| PL Recall | 0.7513 | 0.0983 | 0.5556 | 0.9167 | 0.7429 |
| PFM Recall | 0.5903 | 0.1461 | 0.3846 | 0.8846 | 0.5662 |
| Flaky Recall | 0.7911 | 0.1356 | 0.5556 | 1.0000 | 0.7889 |

## Per-class Recall Summary
| Class | Recall Mean | Recall Std | Recall Min | Recall Max |
|---|---|---|---|---|
| point_like | 0.7513 | 0.0983 | 0.5556 | 0.9167 |
| point_flaky_mixed | 0.5903 | 0.1461 | 0.3846 | 0.8846 |
| flaky | 0.7911 | 0.1356 | 0.5556 | 1.0000 |

## Fold-wise Results
| Fold | Val BA | Test BA | Test Macro F1 | Accuracy | PL Recall | PFM Recall | Flaky Recall | Best Epoch | Notes |
|---|---|---|---|---|---|---|---|---|---|
| fold_00 | 0.7198 | 0.7500 | 0.7062 | 0.7639 | 0.9167 | 0.5556 | 0.7778 | 6 |  |
| fold_01 | 0.7756 | 0.7588 | 0.7249 | 0.7639 | 0.8611 | 0.6154 | 0.8000 | 6 |  |
| fold_02 | 0.7130 | 0.6467 | 0.5847 | 0.6338 | 0.7778 | 0.3846 | 0.7778 | 6 |  |
| fold_03 | 0.6667 | 0.8597 | 0.7794 | 0.8028 | 0.6944 | 0.8846 | 1.0000 | 6 |  |
| fold_04 | 0.7009 | 0.7293 | 0.6572 | 0.6901 | 0.7222 | 0.5769 | 0.8889 | 5 |  |
| fold_05 | 0.7308 | 0.7422 | 0.6997 | 0.7465 | 0.8333 | 0.6154 | 0.7778 | 5 |  |
| fold_06 | 0.7358 | 0.6766 | 0.6510 | 0.7042 | 0.6667 | 0.8077 | 0.5556 | 4 |  |
| fold_07 | 0.6854 | 0.5370 | 0.4996 | 0.5352 | 0.5556 | 0.5000 | 0.5556 | 3 |  |
| fold_08 | 0.7414 | 0.6921 | 0.6120 | 0.6479 | 0.7429 | 0.4444 | 0.8889 | 5 |  |
| fold_09 | 0.7191 | 0.7168 | 0.6404 | 0.6761 | 0.7429 | 0.5185 | 0.8889 | 5 |  |