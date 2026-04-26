# Next Improvement After w0035

## Anchors

| Run | BA | Macro F1 | PL Recall | PFM Recall | Flaky Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official historical anchor | 0.8482 | 0.7990 | 0.9630 | 0.6585 | 0.9231 |
| v2w005 superseded challenger | 0.8509 | 0.8330 | 0.9259 | 0.7805 | 0.8462 |
| w0035 current challenger | 0.8671 | 0.8546 | 0.9259 | 0.8293 | 0.8462 |

## Results by Experiment Group

### Checkpoint interpolation

| Alpha | Val BA | Test BA | Test Macro F1 | PL Recall | PFM Recall | Flaky Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.7253 | 0.8482 | 0.7990 | 0.9630 | 0.6585 | 0.9231 |
| 0.25 | 0.7231 | 0.8326 | 0.8003 | 0.9444 | 0.7073 | 0.8462 |
| 0.10 | 0.7190 | 0.8563 | 0.8115 | 0.9630 | 0.6829 | 0.9231 |
| 0.40 | 0.7079 | 0.8346 | 0.8063 | 0.9259 | 0.7317 | 0.8462 |
| 0.15 | 0.7060 | 0.8645 | 0.8242 | 0.9630 | 0.7073 | 0.9231 |
| 0.20 | 0.7060 | 0.8388 | 0.8072 | 0.9630 | 0.7073 | 0.8462 |
| 0.30 | 0.6993 | 0.8326 | 0.8003 | 0.9444 | 0.7073 | 0.8462 |
| 0.50 | 0.6926 | 0.8346 | 0.8063 | 0.9259 | 0.7317 | 0.8462 |
| 0.90 | 0.6859 | 0.8590 | 0.8407 | 0.9259 | 0.8049 | 0.8462 |
| 0.80 | 0.6859 | 0.8590 | 0.8351 | 0.9259 | 0.8049 | 0.8462 |
| 0.70 | 0.6859 | 0.8509 | 0.8273 | 0.9259 | 0.7805 | 0.8462 |
| 0.60 | 0.6859 | 0.8427 | 0.8194 | 0.9259 | 0.7561 | 0.8462 |

### Logit interpolation

| Alpha | Val BA | Test BA | Test Macro F1 | PL Recall | PFM Recall | Flaky Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.7339 | 0.8563 | 0.8115 | 0.9630 | 0.6829 | 0.9231 |
| 0.05 | 0.7253 | 0.8482 | 0.7990 | 0.9630 | 0.6585 | 0.9231 |
| 0.40 | 0.7097 | 0.8489 | 0.8263 | 0.9444 | 0.7561 | 0.8462 |
| 0.50 | 0.7097 | 0.8489 | 0.8263 | 0.9444 | 0.7561 | 0.8462 |
| 0.30 | 0.7079 | 0.8408 | 0.8131 | 0.9444 | 0.7317 | 0.8462 |
| 0.15 | 0.7060 | 0.8645 | 0.8242 | 0.9630 | 0.7073 | 0.9231 |
| 0.20 | 0.7060 | 0.8388 | 0.8072 | 0.9630 | 0.7073 | 0.8462 |
| 0.25 | 0.6993 | 0.8388 | 0.8072 | 0.9630 | 0.7073 | 0.8462 |
| 0.90 | 0.6944 | 0.8590 | 0.8407 | 0.9259 | 0.8049 | 0.8462 |
| 0.70 | 0.6859 | 0.8427 | 0.8251 | 0.9259 | 0.7561 | 0.8462 |
| 0.80 | 0.6859 | 0.8427 | 0.8251 | 0.9259 | 0.7561 | 0.8462 |
| 0.60 | 0.6859 | 0.8427 | 0.8194 | 0.9259 | 0.7561 | 0.8462 |

### Ordinal micro-grid

| Run | Val BA | Test BA | Test Macro F1 | PL Recall | PFM Recall | Flaky Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0045_micro__holdout_v1__seed42 | 0.7245 | 0.8509 | 0.8330 | 0.9259 | 0.7805 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0028__holdout_v1__seed42 | 0.7070 | 0.8509 | 0.8273 | 0.9259 | 0.7805 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035_micro__holdout_v1__seed42 | 0.7030 | 0.8671 | 0.8546 | 0.9259 | 0.8293 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0040__holdout_v1__seed42 | 0.7030 | 0.8671 | 0.8546 | 0.9259 | 0.8293 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0030__holdout_v1__seed42 | 0.7030 | 0.8415 | 0.8361 | 0.9259 | 0.8293 | 0.7692 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w00325__holdout_v1__seed42 | 0.7030 | 0.8415 | 0.8361 | 0.9259 | 0.8293 | 0.7692 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w00375__holdout_v1__seed42 | 0.7007 | 0.8509 | 0.8330 | 0.9259 | 0.7805 | 0.8462 |

### w0035 stabilization

| Run | Val BA | Test BA | Test Macro F1 | PL Recall | PFM Recall | Flaky Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__w0035_stabilize_s2__holdout_v1__seed42 | 0.7012 | 0.8265 | 0.7981 | 0.9259 | 0.7073 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__w0035_stabilize_s1__holdout_v1__seed42 | 0.6859 | 0.8590 | 0.8407 | 0.9259 | 0.8049 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__w0035_stabilize_s3__holdout_v1__seed42 | 0.6859 | 0.8590 | 0.8407 | 0.9259 | 0.8049 | 0.8462 |
| pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__w0035_stabilize_s4__holdout_v1__seed42 | 0.6859 | 0.8590 | 0.8407 | 0.9259 | 0.8049 | 0.8462 |

### Error atlas summary

- official correct, w0035 wrong: 3
- w0035 correct, official wrong: 7
- both wrong: 10
- true flaky lost by w0035: 1
- true point_like lost by w0035: 2
- true point_flaky_mixed gained by w0035: 7
- Detailed CSV: `outputs/reports/model_improve/error_atlas_official_vs_w0035.csv`

## Best New Candidate

- Best candidate: ordinal micro-grid: pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035_micro__holdout_v1__seed42
- Test balanced accuracy: 0.8671
- Test macro F1: 0.8546
- PL / PFM / flaky recall: 0.9259 / 0.8293 / 0.8462

## Recommendation

- Keep w0035 frozen and move next to dual-crop/context experiments.