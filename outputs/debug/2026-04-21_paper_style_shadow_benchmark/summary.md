# Paper-Style Shadow Benchmark Summary

## Labeling
This pass is a paper-style shadow benchmark only. It is diagnostic and not directly comparable to the official leakage-safe holdout leaderboard.

## Implemented shadow protocol
- Preprocessing: `diagnostics2024_proxy_preproc_v1`
- Split protocol: one non-group-aware stratified 70/15/15 shadow split (`diagnostics2024_shadow_v1`) instead of the repo's duplicate-aware holdout
- Augmentation: `diagnostics2024_shadow_v1`
- Models run: ViT small patch16 224 and ConvNeXtV2 Tiny for scenarios 1 to 3
- Not run: optional stronger transformer, to keep the pass bounded

## Key results
- Scenario 1 (`pattern_3class`): ViT BA 0.6857, ConvNeXt BA 0.3395, winner `vit_small_patch16_224`.
- Scenario 2 (`task_tg_5class`): ViT BA 0.3407, ConvNeXt BA 0.2000, winner `vit_small_patch16_224`.
- Scenario 3 (`severity_5class`): ViT BA 0.3817, ConvNeXt BA 0.3768, winner `vit_small_patch16_224`.

## Interpretation
- The shadow proxy did not reproduce a score jump relative to the strict references. In this approximation, the loose split plus paper-style preprocessing was not enough to explain the published-looking score gap.
- ViT beat ConvNeXtV2 Tiny in all three shadow scenarios. Under this proxy stack, backbone choice still mattered, and the repo's standard ConvNeXt recipe did not transfer cleanly to the grayscale/Otsu masking regime.
- TG remained dominated by `patch_gt_1mm`; the punctate family stayed weak and `coalescent_macro_punctate` recall remained at 0.0 in the best shadow TG run.
- Severity improved modestly relative to the weak post-hoc salvage branch but still did not approach the best strict structured severity reference.

## Artifacts
- `shadow_experiment_summary.csv` contains per-run metrics and recalls.
- `scenario_comparison.csv` contains shadow ViT vs shadow ConvNeXt comparisons.
- `official_vs_shadow_comparison.csv` anchors shadow winners against the current strict references for interpretation only.
