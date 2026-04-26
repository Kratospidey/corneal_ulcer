# Post-Merge Preservation Note

## Status

The param branch containing the w0035 generated challenger state has been merged into main.

A fresh param branch has been recreated from updated main for future experiments.

## Preserved state

Best generated challenger:

pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42

Metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8796 |
| Balanced accuracy | 0.8671 |
| Macro F1 | 0.8546 |
| Weighted F1 | 0.8801 |
| ECE | 0.0728 |

## Git preservation

Old param tip is preserved by:

- archive branch: archive/param-w0035-20260426
- tag: w0035-param-preserved-20260426

## Local ignored artifacts

The checkpoint is local and ignored by Git:

models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42/best.pt

The friend-share figure bundle is local:

outputs/friend_share/w0035_best_challenger_figures_bundle.zip

Backup folder:

/home/kratospidey/corneal_w0035_backup_20260426_160250

## Important

GitHub source code, configs, and reports are preserved on main. Large ignored binary artifacts such as best.pt must be preserved separately through backup, release asset, or Git LFS.
