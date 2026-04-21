# Embedding Analysis Summary

- Backbone focus: convnextv2_tiny
- Representations compared: raw_rgb, masked_highlight_proxy
- Embedding artifact rows: 2
- Projection rows: 1424

## Representation summaries

- convnextv2_tiny / masked_highlight_proxy: method=umap, pattern_mismatch=0.462079, severity_mismatch=0.675562, tg_mismatch=0.36236, mean_neighbor_distance=0.099705
- convnextv2_tiny / raw_rgb: method=umap, pattern_mismatch=0.393258, severity_mismatch=0.682584, tg_mismatch=0.328652, mean_neighbor_distance=0.169312

## Interpretation

- Neighbor mismatch ratios are used here as EDA signals for label mixing, not as model-quality metrics.
- Any apparent separation between raw and preprocessed embeddings should be treated as a hypothesis for Stage 3, not evidence that one path is inherently superior.