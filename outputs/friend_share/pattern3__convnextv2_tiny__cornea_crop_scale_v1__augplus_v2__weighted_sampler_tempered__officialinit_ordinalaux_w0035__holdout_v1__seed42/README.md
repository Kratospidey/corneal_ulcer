# w0035 Best Challenger Results Bundle

## Model

pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42

## Final Test Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8796 |
| Balanced accuracy | 0.8671 |
| Macro F1 | 0.8546 |
| Weighted F1 | 0.8801 |
| ECE | 0.0728 |

## Per-Class Recall

| Class | Recall |
| --- | ---: |
| point_like | 0.9259 |
| point_flaky_mixed | 0.8293 |
| flaky | 0.8462 |

## Folder Guide

| Folder | Contents |
| --- | --- |
| paper_figures | consolidated paper-ready figures |
| confusion_matrices | confusion matrix and normalized confusion matrix |
| roc_curves | ROC curves |
| pr_curves | precision-recall curves |
| explainability | Grad-CAM / XAI gallery |
| model_snapshot | config, metrics, summary, checksums |
| reports | freeze handoff and improvement summary |

## Notes

The model checkpoint best.pt is not included in this share bundle because exported model files are local/ignored in this repository.

This bundle intentionally excludes the layer-wise feature-probe table. That analysis used shallow probes on frozen intermediate features and should not be confused with the final model performance above.
