# External Signal Results

Date: `2026-04-24`

## Scope

Executed and verified external-signal rows:

- current-branch canonical ConvNeXtV2 control
- current-branch `SLID` cornea-mask warm-start
- current-branch `SLIT-Net` white-light 7-class warm-start

Frozen authoritative references remain unchanged:

- official single-model benchmark:
  - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - frozen test BA: `0.8482`
  - frozen test macro F1: `0.7990`
- deployed late-fusion rule:
  - `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
  - frozen test BA: `0.8563`
  - frozen test macro F1: `0.8115`

Important source note:

- frozen truth docs do not record validation metrics for the two frozen reference rows
- validation metrics shown below for those rows come from current local artifacts only

## What Was Actually Usable

### `SLID`

Usable role:

- external slit-lamp anatomy warm-start through binary cornea segmentation

Reused assets:

- `data/external/slid/Original_Slit-lamp_Images.zip`
- `data/interim/slid/manifest.csv`
- `data/interim/slid/cornea_masks/`
- `data/interim/slid/split_files/slid_cornea_pretrain_holdout.csv`

Executed upstream lineage:

- pretrain experiment:
  - `pretrain__slid__convnextv2_tiny__cornea_mask__seed42`
- exported warm-start source:
  - `models/exported/pretrain__slid__convnextv2_tiny__cornea_mask__seed42/best.pt`

### `SLIT-Net`

Usable role:

- one narrow upstream warm-start task only

Executed upstream lineage:

- pretrain experiment:
  - `pretrain__slitnet__convnextv2_tiny__white7_fold1__seed42`
- exported warm-start source:
  - `models/exported/pretrain__slitnet__convnextv2_tiny__white7_fold1__seed42/best.pt`
- verified task:
  - `white_light_7class_segmentation`
- verified fold protocol:
  - `fold1_train_k2-k6_val_k7_test_k1`

Historical context only:

- public Duke Box links failed on `2026-04-23`
- this did not block the current run because the local payload was later provided and audited directly on disk

## Initialization Chains

`SLID` line:

1. ImageNet-pretrained `convnextv2_tiny`
2. `SLID` cornea-mask pretraining
3. SUSTech `pattern_3class` fine-tune with a fresh classifier head

`SLIT-Net` line:

1. ImageNet-pretrained `convnextv2_tiny`
2. `SLIT-Net` white-light 7-class pretraining
3. SUSTech `pattern_3class` fine-tune with a fresh classifier head

## 5-Way Comparison

| Line | Exact experiment | Upstream warm-start source | Val BA | Val Macro F1 | Test BA | Test Macro F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| frozen official single-model benchmark | `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42` | none | `0.6916` | `0.6606` | `0.8482` | `0.7990` |
| frozen deployed late-fusion rule | `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo` | n/a | `0.7128` | `0.6867` | `0.8563` | `0.8115` |
| current-branch control | `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42__currentbranch_control` | none | `0.6710` | `0.6697` | `0.7518` | `0.7657` |
| current-branch `SLID` warm-start | `pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42` | `models/exported/pretrain__slid__convnextv2_tiny__cornea_mask__seed42/best.pt` | `0.6823` | `0.6832` | `0.8074` | `0.7940` |
| current-branch `SLIT-Net` warm-start | `pattern3__convnextv2_tiny__external_slitnet_white7_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42` | `models/exported/pretrain__slitnet__convnextv2_tiny__white7_fold1__seed42/best.pt` | `0.6997` | `0.6833` | `0.6095` | `0.6001` |

## Verified SLIT-Net Row Details

Upstream checkpoint path:

- `models/exported/pretrain__slitnet__convnextv2_tiny__white7_fold1__seed42/best.pt`

Downstream checkpoint path:

- `models/exported/pattern3__convnextv2_tiny__external_slitnet_white7_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt`

Metrics paths:

- `outputs/metrics/pattern3__convnextv2_tiny__external_slitnet_white7_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/val_metrics.json`
- `outputs/metrics/pattern3__convnextv2_tiny__external_slitnet_white7_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/test_metrics.json`

Prediction export paths:

- `outputs/predictions/pattern3__convnextv2_tiny__external_slitnet_white7_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/val_predictions.csv`
- `outputs/predictions/pattern3__convnextv2_tiny__external_slitnet_white7_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/test_predictions.csv`

Report paths:

- `outputs/reports/pattern3__convnextv2_tiny__external_slitnet_white7_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/run_metadata.json`
- `outputs/reports/pattern3__convnextv2_tiny__external_slitnet_white7_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/val_summary.md`
- `outputs/reports/pattern3__convnextv2_tiny__external_slitnet_white7_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/test_summary.md`

## Delta Analysis

### `SLID` Warm-Start Vs Current-Branch Control

- val BA delta: `+0.0113`
- val macro F1 delta: `+0.0135`
- test BA delta: `+0.0556`
- test macro F1 delta: `+0.0283`

### `SLIT-Net` Warm-Start Vs Current-Branch Control

- val BA delta: `+0.0287`
- val macro F1 delta: `+0.0136`
- test BA delta: `-0.1423`
- test macro F1 delta: `-0.1657`

### `SLIT-Net` Warm-Start Vs `SLID` Warm-Start

- val BA delta: `+0.0175`
- val macro F1 delta: `+0.0001`
- test BA delta: `-0.1979`
- test macro F1 delta: `-0.1940`

### `SLIT-Net` Warm-Start Vs Frozen Official Single-Model Benchmark

- test BA delta: `-0.2387`
- test macro F1 delta: `-0.1989`

### `SLIT-Net` Warm-Start Vs Frozen Deployed Late-Fusion Rule

- test BA delta: `-0.2468`
- test macro F1 delta: `-0.2114`

## Verdict

`SLID`:

- materially helped relative to the current-branch control
- did not beat frozen official single-model truth
- did not reach the deployed late-fusion rule

`SLIT-Net`:

- improved validation balanced accuracy relative to both the current-branch control and the `SLID` line
- did not hold up on the test split
- underperformed the current-branch control on test
- underperformed the `SLID` warm-start on test
- remained far below both frozen project references

Promotion status:

- neither external-signal line is promotable over frozen truth
- `SLID` remains the only external-signal path that showed a useful branch-local gain
- `SLIT-Net` is a negative transfer result in the current narrow setup

## Downstream Invariance And Leakage Checks

Verified for the `SLIT-Net` downstream run:

- task remained `pattern_3class`
- backbone remained `convnextv2_tiny`
- preprocessing remained `cornea_crop_scale_v1`
- train transforms remained `pattern_augplus_v2`
- sampler remained `weighted_sampler_tempered`
- no TG/type codepath was used
- no severity codepath was used
- no external-data relabeling of SUSTech targets was introduced
- warm-start lineage was recorded in `run_metadata.json`
- downstream artifact is current-branch output, not a historical checkpoint reused as the claimed result

Canonical downstream split remained:

- train `498`
- val `106`
- test `108`

Cross-dataset contamination checks remained:

- distinct path namespaces
- basename overlap count `0`
- exact-file SHA-256 overlap count `0`

## Config And Command Lineage

Verified SLIT-Net config lineage:

- upstream config:
  - `configs/pretrain_slitnet_convnextv2_tiny_white7_fold1.yaml`
- downstream config:
  - `configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__from_slitnet_white7_pretrain.yaml`

Verified downstream fine-tune command from the completed run:

```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python \
  src/main_train.py \
  --config configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__from_slitnet_white7_pretrain.yaml \
  --device cuda
```

Upstream pretrain note:

- upstream `run_context.json` records the exact config path and experiment lineage
- it does not record the full shell argv
- the executed upstream stage is therefore attributed by verified script/config lineage, not reconstructed shell history
