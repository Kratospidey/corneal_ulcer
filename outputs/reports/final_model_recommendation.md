# Final Model Recommendation

- Best ConvNeXtV2 pattern run: pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_valtuned__holdout_v1__seed42
- Balanced accuracy: 0.8264656150835012
- Macro F1: 0.7943071496153689
- Baseline reference: pattern3__alexnet__raw_rgb__holdout_v1__seed42
- Balanced accuracy delta vs baseline: +0.1082
- Macro F1 delta vs baseline: +0.1127
- Recommendation: promote ConvNeXtV2 as the official final model family for the pattern 3-class task.
- Raw RGB remains the default path unless a matched masked_highlight_proxy run shows a real holdout gain.
- Ranking is based on balanced accuracy first, then macro F1, then per-class recall review.
- These recommendations summarize computed project results only.