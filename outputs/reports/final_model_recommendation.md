# Final Model Recommendation

- Best ConvNeXtV2 pattern run: pattern3__convnextv2_tiny__raw_rgb__holdout_v1__seed42
- Balanced accuracy: 0.6963032450837329
- Macro F1: 0.7006698564593301
- Baseline reference: pattern3__alexnet__raw_rgb__holdout_v1__seed42
- Balanced accuracy delta vs baseline: -0.0220
- Macro F1 delta vs baseline: +0.0191
- Recommendation: keep the Stage 3 AlexNet raw baseline as the official reference and treat ConvNeXtV2 as exploratory or task-specific.
- Raw RGB remains the default path unless a matched masked_highlight_proxy run shows a real holdout gain.
- Ranking is based on balanced accuracy first, then macro F1, then per-class recall review.
- These recommendations summarize computed project results only.