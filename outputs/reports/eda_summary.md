# EDA Summary

## Runtime

- Resolved device: cuda
- Torch available: True
- Pillow available: True
- Embedding backbones: convnextv2_tiny
- Embedding comparison variant: masked_highlight_proxy

## Dataset recap

- Manifest rows: 712
- Raw images: 712
- Cornea masks: 712
- Ulcer masks: 354

## Computed outputs

- Image stats rows: 712
- Mean brightness: 60.0989
- Mean contrast: 47.8412
- Mask stats rows: 712
- Duplicate candidate rows: 8
- Embedding summary rows: 2
- Embedding projection rows: 1424

## Interpretation

- This EDA is a classification-readiness pass, not a model benchmark.
- Paper claims are not project results; only on-disk counts and computed artifacts are treated as project findings.
- Raw RGB and one Diagnostics-inspired preprocessing proxy are both preserved for Stage 3 comparison.