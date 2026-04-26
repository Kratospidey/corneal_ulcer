# Dual Crop Context Results

| Run | Init Checkpoint | Input Mode | Fusion Type | Ordinal Weight | Val BA | Val Macro F1 | Test BA | Test Macro F1 | Accuracy | PL Recall | PFM Recall | Flaky Recall | Beats w0035 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w0035 | None | single_crop | None | 0.035 | 0.7030 | 0.7039 | 0.8671 | 0.8546 | 0.8796 | 0.9259 | 0.8293 | 0.8462 | baseline | Current best |
| D0 | official | dual_crop_context_v1 | concat | 0.035 | 0.6926 | 0.6868 | 0.8171 | 0.8013 | 0.8426 | 0.9259 | 0.7561 | 0.7692 | No | Requires higher LR |
| D1 | w0035 | dual_crop_context_v1 | concat | 0.015 | 0.5075 | 0.4852 | 0.6248 | 0.5519 | 0.5833 | 0.4259 | 0.7561 | 0.6923 | No | LR too low to train random head |
| D2 | official | dual_crop_context_v1 | concat | 0.025 | 0.6926 | 0.6868 | 0.8171 | 0.8013 | 0.8426 | 0.9259 | 0.7561 | 0.7692 | No | Almost identical to D0 |
| D3 | w0035 | dual_crop_context_v1 | concat | 0.015 | 0.3465 | 0.2298 | 0.4252 | 0.2631 | 0.2870 | 0.0185 | 0.4878 | 0.7692 | No | Frozen backbone + low LR = underfit |
| D4 | official | dual_crop_context_v1 | concat | 0.035 | 0.6994 | 0.6991 | 0.7061 | 0.7118 | 0.7870 | 0.8519 | 0.8049 | 0.4615 | No | High LR improved val but overfit/degraded test |