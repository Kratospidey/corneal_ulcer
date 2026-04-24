# ConvNeXtV2 vs Baseline Summary

- Best ConvNeXtV2 pattern run by ranking: pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_valtuned__holdout_v1__seed42
- Official Stage 3 baseline reference: pattern3__alexnet__raw_rgb__holdout_v1__seed42
- Balanced accuracy delta vs baseline: +0.1082
- Macro F1 delta vs baseline: +0.1127
- Baseline reference metrics: balanced_accuracy=0.718272994695759, macro_f1=0.681608005521049

## Raw vs Masked Comparison

- convnextv2_base: only raw_rgb run is available.
- convnextv2_tiny: raw_rgb=0.6963032450837329 vs masked_highlight_proxy=0.5274708729993284 (delta=-0.1688)

## Stronger Variant Check

- Stronger variant comparison: pattern3__convnextv2_base__raw_rgb__holdout_v1__seed42 vs pattern3__convnextv2_tiny__raw_rgb__holdout_v1__seed42
- Balanced accuracy delta: -0.1273
- Macro F1 delta: -0.1306
- Cost proxy: stronger checkpoint=1003.97 MB batch_size=8 vs tiny checkpoint=319.13 MB batch_size=16