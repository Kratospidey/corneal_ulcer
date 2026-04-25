# Error Atlas: Official vs w0035

## Group Counts

- official correct, w0035 wrong: 3
- w0035 correct, official wrong: 7
- both wrong: 10
- both correct: 88

## Targeted Summaries

- true flaky lost by w0035: 1
- true point_like lost by w0035: 2
- true point_flaky_mixed gained by w0035: 7

## Changed Or Wrong Cases

| image_id | group | case type | true | official | w0035 | official conf | w0035 conf | thumbnail |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| 1 | official_correct_w0035_wrong | true_point_like_lost_by_w0035 | point_like | point_like | point_flaky_mixed | 0.5620 | 0.9315 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/official_correct_w0035_wrong/1.png |
| 106 | both_wrong | other | point_like | flaky | flaky | 0.8803 | 0.9410 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/both_wrong/106.png |
| 127 | official_correct_w0035_wrong | true_point_like_lost_by_w0035 | point_like | point_like | point_flaky_mixed | 0.8690 | 0.8339 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/official_correct_w0035_wrong/127.png |
| 264 | both_wrong | other | point_like | flaky | flaky | 0.5308 | 0.7428 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/both_wrong/264.png |
| 391 | both_wrong | true_pfm_unresolved | point_flaky_mixed | flaky | point_like | 0.6594 | 0.5613 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/both_wrong/391.png |
| 400 | both_wrong | true_pfm_unresolved | point_flaky_mixed | flaky | flaky | 0.9619 | 0.8811 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/both_wrong/400.png |
| 402 | both_wrong | true_pfm_unresolved | point_flaky_mixed | flaky | flaky | 0.4784 | 0.5274 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/both_wrong/402.png |
| 421 | w0035_correct_official_wrong | true_pfm_gained_by_w0035 | point_flaky_mixed | flaky | point_flaky_mixed | 0.5577 | 0.9645 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/w0035_correct_official_wrong/421.png |
| 463 | w0035_correct_official_wrong | true_pfm_gained_by_w0035 | point_flaky_mixed | flaky | point_flaky_mixed | 0.9344 | 0.5320 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/w0035_correct_official_wrong/463.png |
| 482 | both_wrong | true_pfm_unresolved | point_flaky_mixed | point_like | point_like | 0.9954 | 0.9432 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/both_wrong/482.png |
| 496 | both_wrong | true_pfm_unresolved | point_flaky_mixed | point_like | point_like | 0.9712 | 0.8613 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/both_wrong/496.png |
| 498 | both_wrong | true_pfm_unresolved | point_flaky_mixed | point_like | point_like | 0.9880 | 0.8161 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/both_wrong/498.png |
| 544 | w0035_correct_official_wrong | true_pfm_gained_by_w0035 | point_flaky_mixed | point_like | point_flaky_mixed | 0.9616 | 0.6375 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/w0035_correct_official_wrong/544.png |
| 574 | w0035_correct_official_wrong | true_pfm_gained_by_w0035 | point_flaky_mixed | point_like | point_flaky_mixed | 0.9830 | 0.7331 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/w0035_correct_official_wrong/574.png |
| 576 | both_wrong | true_pfm_unresolved | point_flaky_mixed | point_like | point_like | 0.9195 | 0.7442 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/both_wrong/576.png |
| 587 | w0035_correct_official_wrong | true_pfm_gained_by_w0035 | point_flaky_mixed | flaky | point_flaky_mixed | 0.5372 | 0.7896 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/w0035_correct_official_wrong/587.png |
| 613 | w0035_correct_official_wrong | true_pfm_gained_by_w0035 | point_flaky_mixed | flaky | point_flaky_mixed | 0.7266 | 0.9346 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/w0035_correct_official_wrong/613.png |
| 620 | w0035_correct_official_wrong | true_pfm_gained_by_w0035 | point_flaky_mixed | flaky | point_flaky_mixed | 0.6304 | 0.7239 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/w0035_correct_official_wrong/620.png |
| 624 | both_wrong | true_flaky_still_wrong_under_w0035 | flaky | point_flaky_mixed | point_flaky_mixed | 0.9871 | 0.9893 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/both_wrong/624.png |
| 682 | official_correct_w0035_wrong | true_flaky_lost_by_w0035 | flaky | flaky | point_flaky_mixed | 0.6585 | 0.9723 | outputs/reports/model_improve/error_atlas_official_vs_w0035_thumbnails/official_correct_w0035_wrong/682.png |