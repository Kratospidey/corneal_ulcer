# Split Recommendations

## Defaults

- Build splits at the raw-image ID level.
- Keep all masks and overlays with the same split as the parent image.
- Prefer duplicate-aware repeated stratified splits or grouped cross-validation.
- Do not finalize patient-aware claims because patient IDs are absent.
- Apply augmentation only after split assignment.

## Stage 3 alignment

- Use raw RGB + convnextv2_tiny as the primary classification-aligned baseline path.
- Use masked_highlight_proxy as the matched paper-inspired comparison path.

## Higher-risk tasks

- The TG task is highly imbalanced and should use stronger imbalance-aware evaluation.
- Any segmentation-assisted classification should be treated as a subset experiment because ulcer masks exist on only part of the cohort.