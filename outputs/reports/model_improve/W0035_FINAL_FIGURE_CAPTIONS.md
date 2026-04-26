# w0035 Final Figure Captions

## Figure 1: Complete System Workflow
Overview of the corneal ulcer classification system. The pipeline processes raw slit-lamp images with cornea masks through a deterministic cropping step (`cornea_crop_scale_v1`), applies stochastic augmentations during training, and feeds the 224x224 tensors to a ConvNeXtV2 Tiny backbone. The final w0035 deployment rule uses a fixed-holdout validation strategy to select the optimal training epoch.

## Figure 2: Preprocessing and Augmentation Pipeline
Visual demonstration of the image preprocessing and augmentation stages. Left to right: the original slit-lamp image, the manually verified cornea mask, the extracted cornea crop (`cornea_crop_scale_v1`), and three independent stochastic augmentation views (`pattern_augplus_v2`). The augmentations include random resizing, rotation, color jitter, and mild Gaussian blur to simulate acquisition variance.

## Figure 3: ConvNeXtV2 Tiny w0035 Architecture
Detailed schematic of the ConvNeXtV2 Tiny architecture employed for the 3-class pattern classification task. The network consists of a stem convolution and four hierarchical feature stages with decreasing spatial dimensions and increasing channel capacities. During training, an auxiliary ordinal loss head (weight 0.035) enforces disease severity progression constraints. The deployment model relies solely on the primary 3-class linear head.

## Figure 4: Training and Validation Selection Curves
Training trajectory of the w0035 model. Panel A plots the primary training and validation cross-entropy loss over epochs. Panels B and C track the validation balanced accuracy and macro F1 scores used for checkpoint selection. The best epoch (vertical dashed line) was chosen purely on validation balanced accuracy prior to any test-set evaluation.

## Figure 5: Evaluation Metrics Panel
Comprehensive evaluation on the frozen holdout test split (n=108). The panel includes the absolute confusion matrix (A), normalized confusion matrix (B), One-vs-Rest Receiver Operating Characteristic (ROC) curves (C), Precision-Recall (PR) curves (D), and empirical calibration curves (E). The model achieves a final Test Balanced Accuracy of 0.8671.

## Figure 6: Grad-CAM XAI Summary
Explainable AI (XAI) overlays using Gradient-weighted Class Activation Mapping (Grad-CAM) from the final convolutional block. The heatmaps highlight regions the network focuses on to make its terminal classification decisions, predominantly aligning with clinically relevant ulcer borders, dense infiltration zones, and peripheral flaky edges.

## Figure 7: Split Sensitivity (10-Fold CV Context)
Robustness analysis of the w0035-style training recipe evaluated across 10 random stratified folds. The CV mean balanced accuracy (0.7109) highlights significant split sensitivity inherent to the small dataset size (N=712). The horizontal red dashed line represents the fixed holdout performance (0.8671), which remains the designated deployment anchor.

## Figure 8: Final Model Story
Development trajectory leading to the w0035 challenger model. The flowchart summarizes the baseline official anchor, the selected w0035 parameters, and the subsequent negative findings from attempted post-recovery strategies (dual-crop, logit soups, and external SSL) that failed to surpass the w0035 baseline's holdout performance.