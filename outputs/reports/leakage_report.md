# Leakage Report

- No official split files were provided by the dataset.
- No patient or visit identifiers are present, so patient-level leakage cannot be ruled out.
- Augmentation must only happen after split assignment.
- Cornea and ulcer masks/overlays are derived assets and must stay with their parent image ID.
- Ulcer masks exist for 354 of 712 images and are label-correlated, so mask presence itself must not become a proxy feature.
- Duplicate and near-duplicate checks found 8 candidate rows, including 4 cross-label suspicions.
- Embedding-neighbor mismatch ratios are summarized for 2 representation/backbone combinations; use them before locking split files.