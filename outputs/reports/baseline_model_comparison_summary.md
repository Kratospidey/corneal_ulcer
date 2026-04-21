# Baseline Model Comparison Summary

- Best pattern run by ranking: pattern3__alexnet__raw_rgb__holdout_v1__seed42
- Most stable pattern run by smallest val/test balanced-accuracy gap: pattern3__alexnet__masked_highlight_proxy__holdout_v1__seed42

## Raw vs Masked Comparison

- alexnet: raw_rgb=0.718272994695759 vs masked_highlight_proxy=0.5151251939869826 (delta=-0.2031)
- resnet18: raw_rgb=0.6277534570217497 vs masked_highlight_proxy=0.48111087948486314 (delta=-0.1466)
- vgg16: raw_rgb=0.7162578463391472 vs masked_highlight_proxy=0.4691705464063188 (delta=-0.2471)

## Severity Extension

- Executed severity run: severity5__alexnet__raw_rgb__holdout_v1__seed42