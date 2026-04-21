# ConvNeXtV2 Error Analysis Summary

## pattern3__convnextv2_tiny__raw_rgb__holdout_v1__seed42

- Confusion: point_flaky_mixed -> point_like count=8
- Confusion: flaky -> point_flaky_mixed count=6
- Confusion: point_flaky_mixed -> flaky count=5
- Lowest-recall class: flaky recall=0.46153846153846156
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
