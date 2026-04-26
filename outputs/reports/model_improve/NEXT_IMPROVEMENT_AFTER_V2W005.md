# Next Improvement After v2w005

## Anchors

| Run | BA | Macro F1 | PL Recall | PFM Recall | Flaky Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official | 0.8482 | 0.7990 | 0.9630 | 0.6585 | 0.9231 |
| v2w005 challenger | 0.8509 | 0.8330 | 0.9259 | 0.7805 | 0.8462 |

## Results by Experiment Group

### Checkpoint interpolation

| Alpha | Val BA | Test BA | Test Macro F1 | PL Recall | PFM Recall | Flaky Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.7276 | 0.8563 | 0.8115 | 0.9630 | 0.6829 | 0.9231 |
| 0.10 | 0.7253 | 0.8482 | 0.7990 | 0.9630 | 0.6585 | 0.9231 |
| 0.60 | 0.7250 | 0.8427 | 0.8143 | 0.9259 | 0.7561 | 0.8462 |
| 0.90 | 0.7245 | 0.8509 | 0.8273 | 0.9259 | 0.7805 | 0.8462 |
| 0.80 | 0.7182 | 0.8427 | 0.8143 | 0.9259 | 0.7561 | 0.8462 |
| 0.30 | 0.7146 | 0.8307 | 0.7947 | 0.9630 | 0.6829 | 0.8462 |
| 0.70 | 0.7097 | 0.8427 | 0.8143 | 0.9259 | 0.7561 | 0.8462 |
| 0.40 | 0.7079 | 0.8326 | 0.8003 | 0.9444 | 0.7073 | 0.8462 |
| 0.50 | 0.7079 | 0.8326 | 0.8003 | 0.9444 | 0.7073 | 0.8462 |

### Logit interpolation

| Alpha | Val BA | Test BA | Test Macro F1 | PL Recall | PFM Recall | Flaky Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.7398 | 0.8408 | 0.8131 | 0.9444 | 0.7317 | 0.8462 |
| 0.80 | 0.7308 | 0.8509 | 0.8273 | 0.9259 | 0.7805 | 0.8462 |
| 0.90 | 0.7308 | 0.8509 | 0.8273 | 0.9259 | 0.7805 | 0.8462 |
| 0.10 | 0.7253 | 0.8563 | 0.8115 | 0.9630 | 0.6829 | 0.9231 |
| 0.70 | 0.7245 | 0.8408 | 0.8131 | 0.9444 | 0.7317 | 0.8462 |
| 0.40 | 0.7227 | 0.8326 | 0.8003 | 0.9444 | 0.7073 | 0.8462 |
| 0.60 | 0.7160 | 0.8408 | 0.8131 | 0.9444 | 0.7317 | 0.8462 |
| 0.20 | 0.7123 | 0.8563 | 0.8115 | 0.9630 | 0.6829 | 0.9231 |
| 0.30 | 0.7056 | 0.8388 | 0.8072 | 0.9630 | 0.7073 | 0.8462 |

### Ordinal weight grid

| Run | Val BA | Test BA | Test Macro F1 | PL Recall | PFM Recall | Flaky Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0045__holdout_v1__seed42 | 0.7245 | 0.8509 | 0.8330 | 0.9259 | 0.7805 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0050__holdout_v1__seed42 | 0.7245 | 0.8509 | 0.8330 | 0.9259 | 0.7805 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0015__holdout_v1__seed42 | 0.7070 | 0.8509 | 0.8273 | 0.9259 | 0.7805 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0025__holdout_v1__seed42 | 0.7070 | 0.8509 | 0.8273 | 0.9259 | 0.7805 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42 | 0.7030 | 0.8671 | 0.8546 | 0.9259 | 0.8293 | 0.8462 |

### Official-teacher distillation

| Run | Val BA | Test BA | Test Macro F1 | PL Recall | PFM Recall | Flaky Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035_distill_w010__holdout_v1__seed42 | 0.7034 | 0.8427 | 0.8194 | 0.9259 | 0.7561 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0050_distill_w010__holdout_v1__seed42 | 0.7034 | 0.8427 | 0.8194 | 0.9259 | 0.7561 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0025_distill_w010__holdout_v1__seed42 | 0.6949 | 0.8427 | 0.8194 | 0.9259 | 0.7561 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0025_distill_w025__holdout_v1__seed42 | 0.6863 | 0.8408 | 0.8131 | 0.9444 | 0.7317 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035_distill_w025__holdout_v1__seed42 | 0.6863 | 0.8265 | 0.7935 | 0.9259 | 0.7073 | 0.8462 |

### Partial freeze

| Run | Val BA | Test BA | Test Macro F1 | PL Recall | PFM Recall | Flaky Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_partialfreeze_p2__holdout_v1__seed42 | 0.7411 | 0.8541 | 0.8124 | 0.9074 | 0.7317 | 0.9231 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_partialfreeze_p1__holdout_v1__seed42 | 0.7348 | 0.8541 | 0.8124 | 0.9074 | 0.7317 | 0.9231 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_partialfreeze_p3__holdout_v1__seed42 | 0.7303 | 0.8420 | 0.7894 | 0.9444 | 0.6585 | 0.9231 |

## Best New Candidate

- Best candidate: ordinal weight grid: pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42
- Test balanced accuracy: 0.8671
- Test macro F1: 0.8546
- PL / PFM / flaky recall: 0.9259 / 0.8293 / 0.8462

## Recommendation

- Promote a new challenger candidate and confirm it with at least one extra seed before any official change.