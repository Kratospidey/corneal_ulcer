# Baseline Error Analysis Summary

## pattern3__alexnet__raw_rgb__holdout_v1__seed42

- Confusion: point_flaky_mixed -> flaky count=11
- Confusion: point_flaky_mixed -> point_like count=8
- Confusion: flaky -> point_flaky_mixed count=4
- Lowest-recall class: point_flaky_mixed recall=0.5365853658536586
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__vgg16__raw_rgb__holdout_v1__seed42

- Confusion: point_flaky_mixed -> flaky count=18
- Confusion: point_like -> point_flaky_mixed count=5
- Confusion: point_like -> flaky count=5
- Lowest-recall class: point_flaky_mixed recall=0.4878048780487805
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__resnet18__raw_rgb__holdout_v1__seed42

- Confusion: point_flaky_mixed -> point_like count=11
- Confusion: flaky -> point_flaky_mixed count=8
- Confusion: point_like -> point_flaky_mixed count=6
- Lowest-recall class: flaky recall=0.38461538461538464
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__alexnet__masked_highlight_proxy__holdout_v1__seed42

- Confusion: point_flaky_mixed -> flaky count=19
- Confusion: point_flaky_mixed -> point_like count=15
- Confusion: point_like -> flaky count=7
- Lowest-recall class: point_flaky_mixed recall=0.17073170731707318
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__resnet18__masked_highlight_proxy__holdout_v1__seed42

- Confusion: point_flaky_mixed -> point_like count=22
- Confusion: point_flaky_mixed -> flaky count=9
- Confusion: point_like -> point_flaky_mixed count=8
- Lowest-recall class: point_flaky_mixed recall=0.24390243902439024
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## pattern3__vgg16__masked_highlight_proxy__holdout_v1__seed42

- Confusion: point_flaky_mixed -> point_like count=18
- Confusion: point_like -> point_flaky_mixed count=15
- Confusion: flaky -> point_flaky_mixed count=8
- Lowest-recall class: flaky recall=0.3076923076923077
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.

## severity5__alexnet__raw_rgb__holdout_v1__seed42

- Confusion: ulcer_leq_50pct -> no_ulcer count=10
- Confusion: ulcer_geq_75pct -> no_ulcer count=10
- Confusion: ulcer_leq_50pct -> ulcer_geq_75pct count=8
- Lowest-recall class: ulcer_leq_50pct recall=0.0967741935483871
- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse.
