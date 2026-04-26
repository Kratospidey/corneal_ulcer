# Prediction Diff Summary

- Samples compared: 108
- Prediction changes: 14
- official correct while A0 wrong: 9
- A0 correct while official wrong: 5

## Per-Class Recall

| Class | official | A0 |
| --- | ---: | ---: |
| point_like | 0.9630 | 0.8889 |
| point_flaky_mixed | 0.6585 | 0.7317 |
| flaky | 0.9231 | 0.6923 |

## point_flaky_mixed Confusion

| Predicted Label | official | A0 |
| --- | ---: | ---: |
| point_like | 6 | 5 |
| point_flaky_mixed | 27 | 30 |
| flaky | 8 | 6 |