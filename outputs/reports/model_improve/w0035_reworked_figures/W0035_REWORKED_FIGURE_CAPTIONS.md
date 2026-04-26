# w0035 Reworked Figure Captions

## Figure 1: w0035 System Pipeline
Overview of the corneal ulcer pattern-classification pipeline. Raw slit-lamp images undergo mask normalization and deterministic cornea-centered cropping before stochastic online augmentation. The ConvNeXtV2 Tiny backbone is trained using a weighted sampler, weighted cross-entropy, and an ordinal auxiliary loss (weight 0.035).

## Figure 2: Preprocessing and Augmentation Examples
Visual demonstration of the image preprocessing and augmentation stages. The pipeline extracts a normalized cornea crop (`cornea_crop_scale_v1`), which is subjected to stochastic augmentations (`pattern_augplus_v2`) during training, including random resizing, rotation, color jitter, and mild Gaussian blur to simulate acquisition variance.

## Figure 3: ConvNeXtV2 w0035 Architecture
Detailed schematic of the ConvNeXtV2 Tiny architecture employed. The network consists of a stem convolution and four hierarchical feature stages. An auxiliary ordinal loss head (weight 0.035) enforces disease severity progression constraints during training, while final deployment relies solely on the primary 3-class linear head.

## Figure 4: Training and Validation Selection Curves
Training trajectory of the w0035 model. Panel A plots the primary training and validation loss over epochs. Panels B and C track the validation balanced accuracy and macro F1 scores. The selected checkpoint (vertical dashed line) was chosen based on validation balanced accuracy prior to test-set evaluation.

## Figure 5: Evaluation Summary
Comprehensive evaluation on the frozen holdout test split. The panel includes absolute and normalized confusion matrices (A, B), One-vs-Rest Receiver Operating Characteristic (ROC) curves (C), Precision-Recall (PR) curves (D), empirical calibration curves (E), and a final metrics summary (F). The model achieves a Test Balanced Accuracy of 0.8671.

## Figure 6: Grad-CAM Summary
Explainable AI (XAI) overlays using Gradient-weighted Class Activation Mapping (Grad-CAM). The heatmaps highlight regions the network focuses on to make its terminal classification decisions, predominantly aligning with clinically relevant ulcer borders and peripheral flaky edges.

## Figure 7: Split Sensitivity (10-Fold CV Context)
Robustness analysis of the w0035-style training recipe evaluated across 10 random stratified folds. The CV mean balanced accuracy (0.7109 ± 0.0795) highlights split sensitivity inherent to the dataset. Cross-validation estimates split robustness and does not replace the fixed-holdout benchmark (dashed line).

## Figure 8: Model Development Summary
Development trajectory leading to the w0035 challenger model. The diagram outlines the path from the official anchor, the selected w0035 parameters, and the subsequent negative findings from attempted post-recovery strategies (dual-crop, logit soups, and external SSL) that failed to surpass the w0035 baseline's holdout performance.