# External Signal Results

Date: `2026-04-23`

## Scope

Executed the approved base external-signal path only:

- external dataset audit
- `SLID` cornea-mask external pretraining
- current-branch canonical ConvNeXtV2 control
- current-branch `SLID` warm-start ConvNeXtV2 fine-tune

Did not execute a fifth row because:

- `SLIT-Net` public data access was not verified in practice
- the base 4-row comparison already produced a clean answer

## What Was Actually Usable

### `SLID`

Usable role:

- external slit-lamp anatomy warm-start through binary cornea segmentation

What was reused:

- `data/external/slid/Original_Slit-lamp_Images.zip`
- `data/interim/slid/manifest.csv`
- `data/interim/slid/cornea_masks/`
- `data/interim/slid/split_files/slid_cornea_pretrain_holdout.csv`

Required repair that was actually performed:

- the manifest pointed to a missing extracted image tree
- the pretraining run extracted `Original_Slit-lamp_Images.zip` into:
  - `data/external/slid/Original_Slit-lamp_Images`

### `SLIT-Net`

Usable role:

- none in this run

Why:

- public code repo was reachable
- advertised dataset and model Box links returned `404` on `2026-04-23`
- repo itself does not contain the dataset payload

## External Pretraining Stage

Executed upstream task:

- binary cornea segmentation from `SLID` cornea masks

Initialization chain:

1. ImageNet-pretrained `convnextv2_tiny`
2. `SLID` cornea-mask external pretraining
3. SUSTech `pattern_3class` fine-tune with a fresh classifier head

External checkpoint lineage:

- checkpoint:
  - `models/checkpoints/pretrain__slid__convnextv2_tiny__cornea_mask__seed42/best.pt`
- exported warm-start source:
  - `models/exported/pretrain__slid__convnextv2_tiny__cornea_mask__seed42/best.pt`
- run context:
  - `outputs/reports/pretrain__slid__convnextv2_tiny__cornea_mask__seed42/run_context.json`

External pretrain summary:

- best epoch: `12`
- best val selection score: `0.9663`
- split support:
  - train `1801`
  - val `386`
  - test `386`

## Executed Comparison Rows

Important source note:

- frozen truth docs remain authoritative for the two project reference test metrics
- frozen docs do not record validation metrics for those rows
- validation metrics for the frozen reference rows below come from the current local artifacts only

| Line | Val BA | Val Macro F1 | Test BA | Test Macro F1 | Test metric source |
| --- | ---: | ---: | ---: | ---: | --- |
| frozen official single-model benchmark | `0.6916` | `0.6606` | `0.8482` | `0.7990` | frozen truth |
| frozen deployed late-fusion rule | `0.7128` | `0.6867` | `0.8563` | `0.8115` | frozen truth |
| current-branch control | `0.6710` | `0.6697` | `0.7518` | `0.7657` | current execution |
| `SLID` warm-start | `0.6823` | `0.6832` | `0.8074` | `0.7940` | current execution |

## Branch-Local Delta

`SLID` warm-start vs current-branch control:

- val BA delta: `+0.0113`
- val macro F1 delta: `+0.0135`
- test BA delta: `+0.0556`
- test macro F1 delta: `+0.0283`

Interpretation:

- in the current branch environment, the external cornea signal helped materially relative to the no-warm-start control

## Delta Vs Frozen Project Baselines

`SLID` warm-start vs frozen official single-model benchmark:

- test BA delta: `-0.0408`
- test macro F1 delta: `-0.0050`

`SLID` warm-start vs frozen deployed late-fusion rule:

- test BA delta: `-0.0489`
- test macro F1 delta: `-0.0175`

## Verdict Against The Predeclared Bar

Result class:

- not yet promotable

Why:

- it did not beat the frozen official single-model benchmark at test balanced accuracy
- it did not reach the deployed late-fusion rule
- branch-local improvement was real, but the absolute result still remained below frozen project truth

Important nuance:

- macro F1 was almost matched to the frozen official single-model benchmark
- the main shortfall remained balanced accuracy relative to frozen truth

## Leakage And Contamination Checks

### Canonical downstream split

Verified:

- SUSTech holdout file remained:
  - `data/interim/split_files/pattern_3class_holdout.csv`
- split counts remained:
  - train `498`
  - val `106`
  - test `108`

### Label definition stability

Verified:

- no change to `pattern_3class` label semantics
- external data was used only as upstream representation signal
- no relabeling of SUSTech targets was introduced

### Model-selection discipline

Verified:

- downstream checkpoint selection stayed on validation `balanced_accuracy`
- no test-based model selection was used

### Cross-dataset contamination checks

Checks performed:

- filename/path namespace check
- duplicate-ID check
- exact-file SHA-256 duplicate check

Results:

- path namespaces were distinct:
  - SUSTech raw images under `data/raw/sustech_sysu/rawImages`
  - `SLID` raw images under `data/external/slid/Original_Slit-lamp_Images`
- basename overlap count: `0`
- exact-file SHA-256 match count across hashed raw images: `0`

Important caveat:

- numeric `image_id` overlap count was high because both datasets use their own integer-like IDs
- this was treated as namespace collision only, not contamination evidence
- the stronger basename/path/hash checks were clean

## Exact Commands Run

External pretraining:

```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python \
  src/pretraining/train_slid_cornea_pretrain.py \
  --config configs/pretrain_slid_convnextv2_tiny_cornea_mask.yaml \
  --device cuda
```

Current-branch control:

```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python \
  src/main_train.py \
  --config configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__currentbranch_control.yaml \
  --device cuda
```

`SLID` warm-start fine-tune:

```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python \
  src/main_train.py \
  --config configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__from_slid_cornea_pretrain.yaml \
  --device cuda
```

## Exact Artifact Paths

External pretrain:

- `models/checkpoints/pretrain__slid__convnextv2_tiny__cornea_mask__seed42/best.pt`
- `models/exported/pretrain__slid__convnextv2_tiny__cornea_mask__seed42/best.pt`
- `outputs/metrics/pretrain__slid__convnextv2_tiny__cornea_mask__seed42/training_summary.json`
- `outputs/reports/pretrain__slid__convnextv2_tiny__cornea_mask__seed42/run_context.json`

Current-branch control:

- `models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42__currentbranch_control/best.pt`
- `outputs/metrics/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42__currentbranch_control/val_metrics.json`
- `outputs/metrics/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42__currentbranch_control/test_metrics.json`
- `outputs/reports/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42__currentbranch_control/run_metadata.json`

`SLID` warm-start:

- `models/exported/pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt`
- `outputs/metrics/pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/val_metrics.json`
- `outputs/metrics/pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/test_metrics.json`
- `outputs/reports/pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/run_metadata.json`

## Historical Context Only

Older local `SLID` warm-start checkpoints were present in the workspace before this run, but they were not treated as final evidence.

They were used only as background context for:

- naming expectations
- prior-art sanity checking
- local artifact discovery

The claimed result above comes only from the current-branch reruns documented in this file.
