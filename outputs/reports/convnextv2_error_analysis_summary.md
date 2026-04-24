# ConvNeXtV2 Error Analysis Summary

## pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_valtuned__holdout_v1__seed42

- Confusion: point_flaky_mixed -> point_like count=7
- Confusion: point_flaky_mixed -> flaky count=5
- Confusion: point_like -> flaky count=3
- Lowest-recall class: point_flaky_mixed recall=0.7073170731707317
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_eq__holdout_v1__seed42

- Confusion: point_flaky_mixed -> point_like count=7
- Confusion: point_flaky_mixed -> flaky count=4
- Confusion: point_like -> flaky count=3
- Lowest-recall class: point_flaky_mixed recall=0.7317073170731707
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42

- Confusion: point_like -> point_flaky_mixed count=5
- Confusion: flaky -> point_flaky_mixed count=4
- Confusion: point_like -> flaky count=3
- Lowest-recall class: flaky recall=0.6923076923076923
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny_plus_swin_tiny__avgprob_eq__holdout_v1__seed42

- Confusion: point_flaky_mixed -> flaky count=9
- Confusion: point_flaky_mixed -> point_like count=6
- Confusion: point_like -> flaky count=3
- Lowest-recall class: point_flaky_mixed recall=0.6341463414634146
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42

- Confusion: point_flaky_mixed -> flaky count=8
- Confusion: point_flaky_mixed -> point_like count=6
- Confusion: point_like -> flaky count=4
- Lowest-recall class: point_flaky_mixed recall=0.6585365853658537
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny_plus_swin_tiny__avgprob_valtuned__holdout_v1__seed42

- Confusion: point_flaky_mixed -> flaky count=8
- Confusion: point_flaky_mixed -> point_like count=6
- Confusion: point_like -> flaky count=4
- Lowest-recall class: point_flaky_mixed recall=0.6585365853658537
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny__slidwarm__image_proxy_geometry_aux__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42

- Confusion: point_flaky_mixed -> point_like count=7
- Confusion: flaky -> point_flaky_mixed count=4
- Confusion: point_like -> flaky count=3
- Lowest-recall class: flaky recall=0.6923076923076923
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo

- Confusion: point_flaky_mixed -> point_like count=8
- Confusion: point_flaky_mixed -> flaky count=7
- Confusion: point_like -> flaky count=3
- Lowest-recall class: point_flaky_mixed recall=0.6341463414634146
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed17

- Confusion: point_flaky_mixed -> point_like count=11
- Confusion: point_like -> point_flaky_mixed count=3
- Confusion: point_like -> flaky count=3
- Lowest-recall class: flaky recall=0.6923076923076923
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed211

- Confusion: point_flaky_mixed -> point_like count=9
- Confusion: point_flaky_mixed -> flaky count=4
- Confusion: flaky -> point_flaky_mixed count=4
- Lowest-recall class: point_flaky_mixed recall=0.6829268292682927
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42__currentbranch_control

- Confusion: point_flaky_mixed -> point_like count=11
- Confusion: flaky -> point_flaky_mixed count=4
- Confusion: point_flaky_mixed -> flaky count=3
- Lowest-recall class: flaky recall=0.6153846153846154
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny_logits_plus_predmaskstats__logreg__holdout_v1__seed42

- Confusion: point_flaky_mixed -> flaky count=10
- Confusion: point_flaky_mixed -> point_like count=9
- Confusion: point_like -> flaky count=5
- Lowest-recall class: point_flaky_mixed recall=0.5365853658536586
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny__raw_rgb__holdout_v1__seed42

- Confusion: point_flaky_mixed -> point_like count=8
- Confusion: flaky -> point_flaky_mixed count=6
- Confusion: point_flaky_mixed -> flaky count=5
- Lowest-recall class: flaky recall=0.46153846153846156
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed101

- Confusion: point_flaky_mixed -> point_like count=12
- Confusion: point_flaky_mixed -> flaky count=7
- Confusion: point_like -> point_flaky_mixed count=4
- Lowest-recall class: point_flaky_mixed recall=0.5365853658536586
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny__external_slitnet_white7_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42

- Confusion: point_flaky_mixed -> flaky count=12
- Confusion: flaky -> point_flaky_mixed count=8
- Confusion: point_flaky_mixed -> point_like count=7
- Lowest-recall class: flaky recall=0.38461538461538464
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed73

- Confusion: point_like -> point_flaky_mixed count=13
- Confusion: point_flaky_mixed -> point_like count=11
- Confusion: point_flaky_mixed -> flaky count=10
- Lowest-recall class: point_flaky_mixed recall=0.4878048780487805
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_base__raw_rgb__holdout_v1__seed42

- Confusion: point_like -> point_flaky_mixed count=11
- Confusion: point_flaky_mixed -> point_like count=9
- Confusion: flaky -> point_flaky_mixed count=9
- Lowest-recall class: flaky recall=0.3076923076923077
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__convnextv2_tiny__masked_highlight_proxy__holdout_v1__seed42

- Confusion: point_flaky_mixed -> point_like count=20
- Confusion: point_flaky_mixed -> flaky count=14
- Confusion: point_like -> point_flaky_mixed count=8
- Lowest-recall class: point_flaky_mixed recall=0.17073170731707318
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.
