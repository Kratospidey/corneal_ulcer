# External SSL Pretraining Summary

## Dataset Audit Summary
The external SLID and SLIT-Net datasets were audited for readable images.
- SLID: 5234 images
- SLIT-Net: 266 images (Blue_Light: 133, White_Light: 133)
- Unreadable files: 0
- Duplicates removed: 0

## External Manifest Summary
A deterministic train/val split (~90/10) was created by hashing the image paths.
- Total images processed: 5500
- `ssl_train` count: 4945
- `ssl_val` count: 555
- Manifest path: `data/external_manifests/slitlamp_external_ssl_manifest.csv`

## SSL Method and Config
- **Method:** SimSiam (Self-Supervised Learning)
- **Encoder:** ConvNeXtV2 Tiny (initialized from ImageNet)
- **Augmentations:** RandomResizedCrop (0.65-1.0), HorizontalFlip, Rotation (15 deg), ColorJitter, GaussianBlur
- **Training:** 30 epochs, AdamW (lr=1e-4, weight_decay=1e-4), CosineAnnealingLR, batch size 16.
- **Config:** `configs/pretrain_external_ssl/convnextv2_tiny_slid_slitnet_simsiam_v1.yaml`

## SSL Pretraining Curves/Metrics
- The SSL model trained stably for 30 epochs.
- Final Train Loss: -0.9662
- Final Val Loss: -0.9771
- Checkpoints saved: `models/pretrained_external/convnextv2_tiny_slid_slitnet_simsiam_v1/backbone.pt` and `ssl_checkpoint.pt`

## Downstream Fine-Tune Results
The pretrained SSL backbone was loaded into the `pattern_3class` recipe, and the classification/ordinal heads were initialized from scratch.
- **SSL-FT1 (Ordinal 0.035):** Test BA 0.6274, Macro F1 0.4450. The model collapsed, predicting mostly `point_like` and `flaky`, almost entirely missing `point_flaky_mixed` (Recall: 0.0488).
- **SSL-FT2 (Ordinal 0.025):** Identical result to FT1.

## Anchor Table

| Run | BA | Macro F1 | Accuracy | PL Recall | PFM Recall | Flaky Recall |
|---|---|---|---|---|---|---|
| Official anchor | 0.8482 | 0.7990 | 0.8426 | 0.9630 | 0.6585 | 0.9231 |
| w0035 current best | 0.8671 | 0.8546 | 0.8796 | 0.9259 | 0.8293 | 0.8462 |
| best external SSL fine-tune (SSL-FT1/FT2) | 0.6274 | 0.4450 | 0.5556 | 0.8333 | 0.0488 | 1.0000 |

## Best Candidate
No external SSL candidate beat the `w0035` baseline.

## 90 Percent BA Target
The 90% BA target was **not** reached. Performance regressed significantly.

## Freeze Recommendation
Do **not** freeze a new challenger. `w0035` remains the best model.

## Next Recommendation
The external SSL pretraining on SLID + SLIT-Net failed to provide a better initialization for the downstream task than the standard supervised recipe starting from the ImageNet-pretrained weights. 

Likely reasons:
- **Domain mismatch:** While SLID/SLIT-Net contain slit-lamp images, the specific feature representations learned by SimSiam on these datasets may not align well with the subtle, fine-grained morphological differences required to distinguish `point_like` from `point_flaky_mixed`.
- **Destructive objective for this task:** SimSiam encourages the model to output the *same* representation for two augmented views of an image. If the augmentations (like crop and blur) destroy the subtle texture cues that define "flaky" vs "point-like," the model learns to ignore those features entirely.
- **Head initialization:** Training the classification head from scratch on only ~500 images while trying to adapt a newly learned representation may be too difficult.

**Next Options:**
- The current architecture and data approach appear saturated. 
- Try anatomy-aware auxiliary pretraining using the region annotations provided in the SLID dataset (e.g., predicting bounding boxes or masks for the cornea/pupil) to force the network to learn relevant spatial features.
- If more performance is strictly required, acquire more human-labeled target data for the specific `pattern_3class` task.