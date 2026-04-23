# External Signal Plan

Date: `2026-04-23`

## Chosen Base Plan

Use `SLID` cornea-mask supervision to warm-start a `convnextv2_tiny` backbone, then fine-tune the canonical `pattern_3class` classifier on the fixed SUSTech holdout recipe.

This is the base execution path because:

- `SLID` is actually accessible in practice
- local `SLID` preparation artifacts exist and are mostly reusable
- the cleanest upstream task is binary cornea segmentation from verified cornea masks
- `SLIT-Net` code is public but the advertised dataset links currently return `404`

## Initialization Chain

Default initialization chain:

1. ImageNet-pretrained `convnextv2_tiny` backbone
2. `SLID` external pretraining on binary cornea segmentation
3. SUSTech `pattern_3class` fine-tune with a freshly initialized classifier head

## Downstream Invariants

The downstream pattern recipe stays fixed:

- task: `pattern_3class`
- split: `data/interim/split_files/pattern_3class_holdout.csv`
- backbone family: `convnextv2_tiny`
- preprocessing: `cornea_crop_scale_v1`
- train augmentation: `pattern_augplus_v2`
- sampler: `weighted_sampler_tempered`
- sampler temperature: `0.65`
- checkpoint selection metric: validation `balanced_accuracy`

## What `SLID` Contributes

Learning signal contributed by `SLID`:

- slit-lamp anatomy prior
- explicit cornea localization prior
- domain-specific feature adaptation before SUSTech pattern fine-tuning

What `SLID` does not contribute in the base run:

- direct pattern label supervision
- direct remapping to SUSTech target classes
- lesion multi-task expansion
- derived weak-label keratitis classification as the main upstream claim

## Local Artifact Reuse Policy

Reused local artifacts:

- `data/external/slid/Original_Slit-lamp_Images.zip`
- `data/interim/slid/manifest.csv`
- `data/interim/slid/cornea_masks/`
- `data/interim/slid/split_files/slid_cornea_pretrain_holdout.csv`

Required repair before execution:

- the current manifest points to an extracted image directory that is missing on disk
- before training, raw-image access must be repaired by either:
  - extracting the zip into the expected directory, or
  - writing a small current-branch path repair / extraction step and documenting it

No historical model checkpoint will be treated as final evidence.

## Executed Rows

The mandatory rows are:

1. frozen official single-model benchmark
2. frozen deployed late-fusion rule
3. current-branch control using the explicit `__currentbranch_control` config
4. current-branch `SLID` warm-start line with explicit external-checkpoint lineage

Optional fifth row:

- only after the 4-row comparison executes cleanly
- only if it is actually justified
- only if it does not delay or bloat the base run
- may be:
  - a `SLIT-Net`-based line if data access becomes real, or
  - one narrowly scoped second `SLID` variant

## Planned Experiment Names

External stage:

- `pretrain__slid__convnextv2_tiny__cornea_mask__seed42`

Current-branch control:

- `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42__currentbranch_control`

Current-branch warm-start:

- `pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`

## Implementation Surface

Minimal intended surface:

- `src/external_data/slid.py`
- `src/pretraining/train_slid_cornea_pretrain.py`
- one explicit warm-start hook in `src/main_train.py`
- one pretrain config
- one control config
- one warm-start config

The existing downstream pattern training and evaluation entrypoints remain the canonical path:

- `src/main_train.py`
- `src/main_eval.py`

## Why This Plan Was Chosen

This plan gives the highest chance of a clean and honest answer to the real project question:

- can external slit-lamp supervision improve the frozen pattern line without new user annotation?

It avoids the recent failure mode of adding more late-stage modeling complexity while keeping the comparison interpretable:

- same downstream recipe
- one upstream signal change
- frozen truth preserved
