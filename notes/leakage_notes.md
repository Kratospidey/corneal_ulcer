# Leakage Notes

## Confirmed leakage and evaluation risks

### No official split definition
- The shipped dataset does not include train/validation/test splits.
- Any later split file must therefore be treated as a project decision, not a dataset-provided ground truth.

### No patient metadata
- No patient ID, eye ID, visit ID, or encounter ID is present.
- Patient-level leakage cannot be ruled out.
- Random image-level splitting may therefore overestimate generalization.

### Augmentation can easily leak
- Diagnostics 2024 reports augmentation by rotation, scaling, padding, and flipping.
- Those operations must happen only after split assignment and only inside the training fold.
- Augmenting first and splitting later would create trivial leakage.

### Preprocessing can leak task information
- Diagnostics 2024 uses a preprocessing chain that includes thresholding and masking.
- If thresholds, masks, or region proposals are derived using information outside the training fold, evaluation can become optimistic.
- Any preprocessing whose parameters are tuned on the full dataset should be treated as leakage-prone.

### Derived assets are not independent samples
- `corneaOverlay`, `ulcerOverlay`, `corneaLabels`, and `ulcerLabels` are derived from the same raw image IDs.
- These files must never be split independently from their parent raw image.
- If a raw image lands in train, all derived assets for that ID must stay in train; same for validation/test.

### Ulcer-mask availability is label-correlated
- Ulcer masks exist for exactly 354 images.
- All observed `Category=1` and `Category=2` images have ulcer masks.
- All observed `Category=0` images do not have ulcer masks.
- This means mask presence itself carries label information.
- Any pipeline that mixes full-cohort raw images with ulcer-mask availability as an implicit feature risks leakage.

### Split instability in the small-data regime
- Bioengineering 2023 uses a 70/30 split on a 712-image dataset.
- Diagnostics 2024 uses 10-fold cross-validation, which is more stable than one holdout but still vulnerable if duplicates or correlated samples cross folds.
- Because the dataset is small and imbalanced for some tasks, naive random splits can produce unstable or overly optimistic results.

## Practical risks for this repository

- The 3-class type task is moderately imbalanced (`358 / 263 / 91`).
- The TG/type-grade task is extremely imbalanced (`36 / 78 / 40 / 10 / 548`).
- The severity task is also imbalanced (`36 / 98 / 203 / 273 / 102`).
- Any split procedure must stratify by the target label and, where possible, group near-duplicates.
- Metrics from a single split should not be treated as definitive.

## Safe defaults for later experiments

- Build splits from raw-image IDs, not from masks or overlays.
- Keep all assets sharing the same numeric ID in the same split.
- Run duplicate and near-duplicate checks before finalizing split files.
- Prefer repeated stratified splits or grouped cross-validation over a single 70/30 holdout.
- Treat segmentation-based experiments as a separate family because ulcer-mask coverage is incomplete and label-correlated.
- Apply augmentation only inside the training partition.
- Document clearly whether an experiment uses:
  - raw images only,
  - cornea masks / cornea crops,
  - ulcer masks / ulcer-derived features,
  - or preprocessing-heavy masked images.

## Recommended interpretation stance

- High reported accuracies in the papers should not be accepted at face value without duplicate-aware and leakage-aware splitting.
- Comparisons between the two papers are not apples-to-apples because one relies on heavy preprocessing and scenario relabeling, while the other uses raw-image transfer features with a fixed holdout split.
- The safest next step for modeling is a duplicate-aware, label-stratified split strategy defined at the raw-image ID level.
