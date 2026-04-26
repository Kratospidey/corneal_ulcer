# Official Checkpoint Lineage Forensics

Date: 2026-04-25

## Question

Can the repo, restored to commit `54facdc17e4ac7f2a04ec62c1a570a26ca18aeb8`, reproduce the official ConvNeXtV2 Tiny crop-scale checkpoint by fresh training, and what does commit `ab1c09d6b1e929bda4a488092f75b301024b3630` actually prove?

## Short Answer

No. The repo can reproducibly evaluate the saved official checkpoint, but the historical tracked code does not reproduce the checkpoint-quality model by fresh training.

Commit `ab1c09d` restored recipe semantics and checkpoint evaluation, but it did not restore fresh-training reproduction. Its own bundled control retrain was already much weaker than the official checkpoint.

## Hard Evidence

### 1. Official checkpoint eval is stable on clean `54facdc`

Using a temporary no-multiprocessing runner config:

- config: `configs/tmp_eval_official_checkpoint_cpu_nw0.yaml`
- checkpoint: `models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt`

Observed metrics:

- val BA: `0.7253`
- val macro F1: `0.6968`
- test BA: `0.8482`
- test macro F1: `0.7990`
- test accuracy: `0.8426`

Artifacts:

- `outputs/tmp_repro_54facdc/reports/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/val_summary.md`
- `outputs/tmp_repro_54facdc/reports/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/test_summary.md`

### 2. Fresh training on clean `54facdc` is weak

Using the official recipe lineage with only `num_workers: 0` added as a shell workaround:

- config: `configs/tmp_train_repro_54facdc_gpu_nw0.yaml`
- experiment: `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__repro54facdc__holdout_v1__seed42`

Observed metrics:

- best val BA: `0.6912` at epoch `2`
- test BA: `0.7416`
- test macro F1: `0.6590`

Artifacts:

- `outputs/repro_54facdc/reports/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__repro54facdc__holdout_v1__seed42/val_summary.md`
- `outputs/repro_54facdc/reports/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__repro54facdc__holdout_v1__seed42/test_summary.md`
- `outputs/repro_54facdc/metrics/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__repro54facdc__holdout_v1__seed42/history.csv`

### 3. `ab1c09d` also failed to reproduce the official checkpoint by fresh training

Commit `ab1c09d` bundled a preserved control rerun:

- experiment suffix: `__currentbranch_control`
- best epoch: `5`
- val BA: `0.6710`
- test BA: `0.7518`

Evidence from commit contents:

- `outputs/metrics/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42__currentbranch_control/val_metrics.json`
- `.../test_metrics.json`
- `.../training_summary.json`
- `.../history.csv`

So `ab1c09d` did not prove fresh-training recovery. It proved recipe/evaluation recovery.

### 4. What `ab1c09d` actually restored

Relevant code changes in `ab1c09d`:

- `src/utils_preprocessing.py`
  - added `normalize_cornea_mask()`
  - restored white-background inversion for cornea masks
  - restored square cornea crop with context and scale clamps
- `src/data/transforms.py`
  - added the exact `pattern_augplus_v2` train augmentation profile
- `configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml`
  - first appearance of the canonical official config in tracked history
- `tests/test_pattern_recipe_regression.py`
  - locked crop geometry and augmentation semantics

This explains why checkpoint evaluation returned to `0.8482`.

### 5. The official checkpoint predates the tracked canonical config

The official checkpoint file timestamp is:

- `2026-04-20 02:28:50 +0530`

But the canonical tracked crop-scale config first appears in:

- `ab1c09d` on `2026-04-24`

Tracked file history also shows:

- `src/utils_preprocessing.py` had no crop-scale path before `ab1c09d`
- `src/data/transforms.py` had no `pattern_augplus_v2` profile before `ab1c09d`
- `configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml` did not exist before `ab1c09d`
- `configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__retrain_lineage_v1.yaml` was only added later in `e04276c`

Implication:

The official winning run was not launched from the currently tracked canonical config on `main`. Its launch surface was either:

- local and untracked at the time, or
- on a deleted/unmerged branch, or
- later reconstructed from memory after the winning artifact already existed

### 6. The official checkpoint payload is insufficient to recover the original lineage

Checkpoint payload keys:

- `epoch`
- `model_state_dict`
- `optimizer_state_dict`

Missing from the checkpoint:

- resolved config
- training history
- validation metrics snapshot
- run metadata
- package versions

Recovered internal facts:

- official checkpoint saved at epoch `6`
- optimizer step count at save time: `188`
- saved learning rate in optimizer state: `7.777851165098013e-05`

By comparison:

- fresh `repro54facdc` best checkpoint saved at epoch `2`
- optimizer step count: `60`

### 7. Training-critical data provenance is not versioned in Git

The files used directly by train/eval are not tracked by Git:

- `data/interim/manifests/manifest.csv`
- `data/interim/split_files/pattern_3class_holdout.csv`

Current machine hashes:

- manifest SHA-256: `a7efafece1924bf38d953e24b7b15ce2aa517d7e3058a5becd12fe950464ec3b`
- split SHA-256: `9e37a668b20a6b9f988ebcfbde9b46824a438cad5aa04ca18445ae8db0d72021`

Implication:

Even with identical tracked Python code, the effective benchmark can drift silently through:

- label edits in `manifest.csv`
- path or mask changes in manifest rows
- split assignment edits in `pattern_3class_holdout.csv`

The current repo does not preserve historical hashes for the data files that were used to create the official checkpoint.

### 8. A later fresh retrain already matched the official validation score

The preserved later run:

- `outputs/retrain_lineage_2026-04-24/...__retrain_lineage_v1__holdout_v1__seed42`

achieved:

- val BA: `0.7268`
- test BA: `0.7710`

This matters because the official checkpoint currently evaluates at:

- val BA: `0.7253`
- test BA: `0.8482`

So the official model's validation checkpoint-selection score has already been matched by a fresh retrain, while the test score was not. That points away from a present-day model implementation bug and toward:

1. small-sample test variance, and/or
2. missing lineage/provenance from the original winner

## Comparison Table

| Run | Val BA | Test BA | Test Macro F1 | Best Epoch / Saved Epoch | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| Official frozen checkpoint | `0.7253` | `0.8482` | `0.7990` | saved at epoch `6` | current eval on clean `54facdc` |
| `ab1c09d` current-branch control rerun | `0.6710` | `0.7518` | `0.7657` | best epoch `5` | bundled in commit |
| `retrain_lineage_v1` fresh retrain | `0.7268` | `0.7710` | `0.7522` | best epoch not preserved here | local preserved run metadata |
| Fresh `repro54facdc` retrain | `0.6912` | `0.7416` | `0.6590` | best epoch `2` | reproduced on clean `54facdc` |

## Conclusion

The tracked repo preserves a strong official artifact, but it does not preserve enough lineage detail to reproduce that artifact by fresh training.

The evidence rules out the simplest explanation:

- it is not a current recipe drift problem
- it is not a later regression from `ab1c09d` to `54facdc`
- it is not clearly a current model implementation bug, because a later fresh retrain already matched the official checkpoint's validation BA on the repo's actual selection metric

Instead, the likely missing factor is provenance:

1. the original winning launch config was not preserved in tracked history
2. the original epoch-by-epoch training history was not preserved
3. the original run metadata and environment snapshot were not preserved
4. the later repo freeze reconstructed the official recipe semantics around an already-existing winner
5. the manifest and split files were never versioned, so data-side drift cannot be ruled out

## Practical Recommendation

Treat the official checkpoint as a frozen anchor artifact, not as a reproducible training lineage, unless older local files or deleted branch history can be recovered from outside the current Git history.

If reproduction remains important, the next best forensic targets are outside the current tracked code:

1. shell history from April 2026
2. old output roots or archived experiment folders on the training machine
3. deleted branches or remote refs not present locally
4. environment/package snapshots from the machine that created the official checkpoint
