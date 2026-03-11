# Paper Grounding Summary

## Diagnostics 2024: ViT on Preprocessed Fluorescein Images

### Task framing
- Uses the SUSTech-SYSU fluorescein-stained corneal image dataset with 712 images.
- Frames the work as three related classification scenarios rather than one fixed task.
- Applies the same overall workflow to all scenarios: label preparation, image preprocessing, augmentation, ViT classification, then metric reporting.

### Scenario definitions
- Scenario 1: 3-class corneal-ulcer type classification.
- Scenario 2: 5-class TG/type-grade classification.
- Scenario 3: 5-class severity classification.

### Class definitions
- Scenario 1 classes:
  - Point-like corneal ulcer: 358 images.
  - Point-flaky mixed corneal ulcer: 263 images.
  - Flaky corneal ulcer: 91 images.
- Scenario 2 classes:
  - No ulcer: 36 images.
  - Micro-punctate ulcers: 78 images.
  - Macro-punctate ulcers: 40 images.
  - Coalescent macro-punctate ulcers: 10 images.
  - Ulcers with patch value greater than 1 mm: 548 images.
- Scenario 3 classes:
  - No ulcer: 36 images.
  - Ulcers affecting no more than 25% of the cornea: 98 images.
  - Ulcers affecting up to 50% of the cornea: 203 images.
  - Ulcers affecting at least 75% of the cornea: 273 images.
  - Ulcers surrounding the center of the cornea: 102 images.

### Preprocessing pipeline
- Removes the blue channel.
- Converts the image to grayscale.
- Applies Gaussian filtering / blurring.
- Applies Otsu thresholding for noise removal.
- Performs masking to keep the information-containing region.
- The paper argues this preprocessing helps isolate diagnostically useful structure before classification.

### Augmentation pipeline
- Rotation.
- Scaling.
- Padding.
- Flipping.
- Reports increasing the dataset from 712 to 3560 images via 5x expansion.
- This augmentation is presented as beneficial, but any reproduction must ensure augmentation is applied only after split assignment.

### Evaluation framing
- Uses Vision Transformer (ViT) as the classifier.
- Reports accuracy, precision, recall, F1-score, and AUC.
- Uses 10-fold cross-validation.
- Reported mean accuracy:
  - Scenario 1: 95.77%.
  - Scenario 2: 96.43%.
  - Scenario 3: 97.27%.

### Important assumptions and weaknesses
- Heavy preprocessing means the learned representation is not directly comparable to a raw-image pipeline.
- The paper treats scenario labels as cleanly available, but in the actual dataset these labels must be reconstructed from a spreadsheet rather than folder names.
- Strong performance gains are partly attributed to augmentation and preprocessing, so leakage control is critical.
- No patient identifiers are available in the public dataset, so patient-level independence cannot be confirmed.
- The paper uses scenario-specific relabeling over the same underlying image set, which makes split hygiene more important than the results tables suggest.

## Bioengineering 2023: Raw-Image Transfer Features with GA + SVM

### Raw-image classification philosophy
- Motivates the work as a small-sample medical-image problem where end-to-end DNN training is fragile.
- Explicitly tries to avoid complex preprocessing and segmented-image dependence.
- Uses raw corneal images directly, aiming to show useful transfer features can still be extracted without a heavy mask-first pipeline.

### Why small dataset size matters
- Argues that deep networks with many parameters are poorly matched to small datasets.
- Treats redundancy in deep feature maps as a major risk when sample size is limited.
- Uses transfer learning to reuse ImageNet-trained representations instead of full training from scratch.

### ResNet-18 / feature extraction / feature selection logic
- Uses pre-trained ResNet-18.
- Extracts feature maps from many intermediate layers rather than relying only on the final pooled layer.
- Evaluates which layer outputs are most useful for the corneal-ulcer task.
- Uses a genetic algorithm (GA) to select a compact subset of effective feature maps and reduce redundancy.
- Notes that deeper late-stage ResNet layers performed better than the classical final pool5 representation in their experiments.

### SVM usage
- Replaces softmax classification with SVM.
- Uses SVM on top of selected ResNet-derived features.
- Claims SVM improves performance relative to the standard softmax head in this small-data setting.

### What avoids heavy fine-tuning
- Uses ResNet as a fixed feature extractor.
- States that the fine-tuning step is eliminated to save time and energy.
- Moves optimization effort into feature-map selection plus SVM classification instead of end-to-end network adaptation.

### Split assumptions
- Uses a 70% training / 30% testing split.
- Reports repeated runs for layer comparison, but the core evaluation framing is still a fixed holdout split rather than cross-validation.

### Methodological caveats
- The paper focuses on ulcer detection/classification from raw images, not the three scenario taxonomy used in Diagnostics 2024.
- Small-sample performance claims can be highly split-sensitive under a single 70/30 partition.
- GA-based feature selection is computationally expensive and potentially unstable if selection is not nested correctly inside training-only data.
- Because it uses raw images, its results are not directly comparable to preprocessing-heavy masked-image pipelines.

### What this implies for EDA
- EDA should examine raw-image separability, not only mask-derived regions.
- Latent-space structure and feature redundancy matter because sample size is small.
- Any claim about representation quality should be conservative and split-aware.
- The dataset should be analyzed as supporting both raw-image and mask/preprocessing-heavy experiment families, not a single unified modeling recipe.

## Synthesis

### Where the papers agree
- The public dataset is small enough that evaluation variance matters.
- Transfer learning is a sensible starting point.
- Corneal-ulcer classification performance depends strongly on representation choice.
- The dataset is suitable for model development, but only with careful interpretation.

### Where they differ
- Diagnostics 2024 uses heavy preprocessing, masking, augmentation, and ViT.
- Bioengineering 2023 uses raw images, fixed ResNet-18 feature extraction, GA-based feature selection, and SVM.
- Diagnostics 2024 frames three scenario tasks from one label system.
- Bioengineering 2023 frames the problem more like raw-image ulcer detection/classification.
- Diagnostics 2024 uses 10-fold cross-validation.
- Bioengineering 2023 uses a 70/30 split.

### What EDA must check because of these differences
- Reconstruct the actual label systems from the spreadsheet instead of assuming paper scenario folders exist.
- Separate raw-image analysis from preprocessing-dependent analysis.
- Verify which assets exist for all images versus only a subset.
- Check whether mask availability is correlated with labels.
- Treat augmentation as a training-only operation and assess leakage risk before any model-readiness claim.
- Prefer duplicate-aware, group-aware split recommendations over naive random splits.
