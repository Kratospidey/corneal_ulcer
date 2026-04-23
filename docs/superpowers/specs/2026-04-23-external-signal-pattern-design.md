# External-Signal Pattern Design

## Goal

Add one clean external-signal continuation path for the frozen pattern-only project by using public slit-lamp supervision upstream, then returning to the canonical `pattern_3class` task for fair evaluation.

The base experiment is:

1. audit real accessibility and structure of `SLID` and `SLIT-Net`
2. reuse existing local `SLID` data-preparation artifacts only if they pass audit
3. run a current-branch ConvNeXtV2 control with no external warm-start
4. run a current-branch ConvNeXtV2 `SLID` warm-start experiment
5. compare both against the frozen official single-model benchmark and the frozen deployed late-fusion rule

This design explicitly excludes TG/type, severity/grade, user-created new annotations, architecture sprawl, and relabeling of the SUSTech target task.

## Frozen Truth To Preserve

- Active task family: `pattern_3class`
- Official single-model benchmark:
  - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - frozen test balanced accuracy `0.8482`
  - frozen test macro F1 `0.7990`
- Best deployed inference rule:
  - `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
  - frozen test balanced accuracy `0.8563`
  - frozen test macro F1 `0.8115`

These frozen references remain authoritative even if local reruns under those paths drift.

## Fixed Downstream Pattern Contract

The external-signal experiment must change only the upstream supervision path. The downstream `pattern_3class` fine-tune stays fixed to the current canonical recipe:

- split file:
  - `data/interim/split_files/pattern_3class_holdout.csv`
- task:
  - `pattern_3class`
- backbone:
  - `convnextv2_tiny`
- preprocessing:
  - `cornea_crop_scale_v1`
- train augmentation:
  - `pattern_augplus_v2`
- sampler:
  - `weighted_sampler_tempered`
- sampler temperature:
  - `0.65`
- checkpoint-selection metric:
  - validation `balanced_accuracy`

No other major modeling change is allowed in the base comparison.

## Chosen Approach

### Default Plan

Use `SLID` as the primary external dataset for anatomy-aware slit-lamp warm-start.

The default upstream task for the base experiment is:

- `SLID` cornea-mask supervision
- one verified supervision target only
- no multi-objective expansion in the base run unless the audit shows that a richer target is both simple and clearly justified

1. audit the existing local `SLID` archive and prepared artifacts
2. verify that `SLID` cornea-mask supervision is actually supported by the audited artifacts
3. train an external pretraining stage that exports ConvNeXtV2 backbone weights
4. fine-tune the canonical pattern classifier on SUSTech using that backbone initialization

### Conditional Extension

Only if `SLIT-Net` is genuinely accessible, licensed clearly enough, and straightforward to integrate without bloating the codebase:

- add one second staged experiment using `SLIT-Net` as an additional auxiliary warm-start or comparison line

### Fallback

If richer anatomy or segmentation supervision is too messy to exploit cleanly:

- use the strongest verified image-level external signal available
- keep the same downstream pattern fine-tune contract

## External Dataset Audit Contract

Before any implementation or training claim, create:

- `docs/superpowers/handoffs/EXTERNAL_SIGNAL_DATASET_AUDIT.md`

For each of `SLID` and `SLIT-Net`, the audit must record:

- verified access status
- source URL or source location
- license / usage terms if present
- modality and image type
- label types available
- total sample count and usable sample count
- on-disk folder or file structure
- recommended role in this repo
- what is usable
- what is not usable
- why it is or is not close enough to support slit-lamp pattern transfer

For each reused `SLID` prepared artifact, the audit must also record:

- exact path
- sample count
- expected schema or expected file layout
- quick integrity-check result

Reused local preparation is allowed only if current-branch code can consume it cleanly and the audit makes the reuse verifiable.

## Warm-Start Contract

The warm-start lineage must be explicit:

- default initialization chain:
  - ImageNet-pretrained `convnextv2_tiny` backbone
  - `SLID` external pretraining on cornea-mask supervision
  - SUSTech `pattern_3class` fine-tune
- the external stage exports ConvNeXtV2 backbone weights
- the downstream `pattern_3class` classifier head is freshly initialized
- any reused historical checkpoint may be inspected for context or sanity checking, but is not acceptable as the final claimed result
- the reported external-signal result must come from a current-branch rerun using current configs

If the implementation needs a warm-start config key, it must point to the exact exported external pretrain checkpoint used to initialize the downstream fine-tune.

## Planned Code Shape

Keep the implementation narrow and namespaced.

Likely additions:

- `src/external_data/...`
  - dataset adapter(s)
  - manifest or metadata parsing helpers
- `src/pretraining/...`
  - one external pretraining entrypoint or helper
- minimal warm-start support in the current mainline pattern path

The existing pattern training and evaluation path should remain the main downstream execution path:

- `src/main_train.py`
- `src/main_eval.py`
- `src/model_factory.py`

The intent is to add one small upstream stage plus a clean weight-loading hook, not a second general-purpose training framework.

## Execution Matrix

The required executed rows are:

1. frozen official single-model benchmark
2. frozen deployed late-fusion rule
3. current-branch canonical ConvNeXtV2 control with no external warm-start
4. current-branch `SLID` warm-start ConvNeXtV2 line

An optional fifth row is allowed only if the base four-row comparison executes cleanly first, and only if it does not delay or bloat the core experiment.

The optional fifth row is allowed only if actually executed and justified:

- `SLIT-Net`-based line, or
- one tightly scoped second `SLID` variant needed for a fair comparison

The base experiment names should stay honest and reproducible. For example:

- external stage:
  - `pretrain__slid__convnextv2_tiny__cornea_mask__seed42`
- downstream control:
  - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42__currentbranch_control`
- downstream warm-start:
  - `pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`

Exact names may be refined during implementation, but they must obey these lineage rules:

- every external checkpoint name must identify the dataset and upstream task
- every downstream fine-tune name must identify that it came from external slit-lamp pretraining
- control and warm-start names must remain visually distinct

## Success Bar

The success bar is fixed before execution:

- promising:
  - test balanced accuracy exceeds `0.8482`
  - macro F1 remains roughly stable and is not clearly worse than frozen `0.7990`
- strong:
  - test balanced accuracy reaches or exceeds `0.8563`
  - no obvious macro-F1 regression against the deployed rule
- not worth keeping:
  - no clear test gain
  - tiny or unstable gain only
  - or balanced-accuracy gain comes with obvious macro-F1 damage

## Stop Rule For Local Drift

If the current-branch canonical no-warm-start control cannot be reproduced cleanly enough to serve as a fair local comparator:

- still compare the new external-signal run against the frozen project references
- report the exact local-control limitation honestly
- mark the mechanistic claim "`SLID` helped in this exact branch environment" as limited rather than definitive

This prevents overclaiming when local artifact drift makes branch-local causality unclear.

## Leakage And Contamination Rules

The implementation and final report must explicitly verify:

- canonical SUSTech holdout split preserved
- no change to the meaning of `pattern_3class` labels
- no external dataset used to relabel SUSTech targets
- no fitting of downstream preprocessors on SUSTech val/test labels
- no test-based model selection
- no train/val/test mixing across downstream pattern evaluation
- no hidden use of TG/type or severity codepaths
- no external examples knowingly overlapping the SUSTech evaluation set without disclosure

If overlap or contamination is unclear and materially concerning, stop and document it.

## Required Deliverables

Create:

- `docs/superpowers/handoffs/EXTERNAL_SIGNAL_DATASET_AUDIT.md`
- `docs/superpowers/handoffs/EXTERNAL_SIGNAL_PLAN.md`
- `docs/superpowers/handoffs/EXTERNAL_SIGNAL_RESULTS.md`

Update only if justified by executed work:

- `README_training.md`
- `codex.md`

## Results Report Contract

`docs/superpowers/handoffs/EXTERNAL_SIGNAL_RESULTS.md` must include:

- exact experiment names
- exact commands run
- exact artifact paths
- exact external pretrain checkpoint path
- exact warm-start source used by the downstream fine-tune config
- val balanced accuracy
- val macro F1
- test balanced accuracy
- test macro F1
- delta vs frozen official single-model benchmark
- delta vs frozen deployed late-fusion rule

Old checkpoint numbers may appear only as historical context and must be labeled clearly as non-authoritative.

## Acceptance Gate

This work is only complete when all of the following are true:

- external dataset accessibility was actually audited
- at least one usable external-signal path was implemented from current-branch code
- at least one real current-branch external-signal experiment was executed
- a current-branch no-warm-start control was executed if possible
- final val/test metrics were produced with exact artifact lineage
- comparison against frozen pattern baselines was honest
- the repo remained pattern-only in scope
