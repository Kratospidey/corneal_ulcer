# External-Signal Pattern Implementation Plan

> `writing-plans` skill fallback: this session does not expose that skill, so this plan is written directly from the approved spec and the repo's existing `docs/superpowers/plans/` convention.

**Goal:** Execute one clean external-signal continuation line for the frozen pattern-only project by auditing `SLID` and `SLIT-Net`, reusing only verified `SLID` artifacts, training one current-branch `SLID` cornea-mask external warm-start, and fine-tuning the canonical ConvNeXtV2 `pattern_3class` line against a current-branch no-warm-start control and the frozen baselines.

**Architecture:** Add a narrow external-data / pretraining path for `SLID` cornea-mask supervision, export a ConvNeXtV2 backbone checkpoint, then reuse the existing `main_train.py` and `main_eval.py` pattern pipeline with one explicit warm-start hook. Keep the downstream recipe fixed: same split, same model family, same crop, same augmentation, same sampler, same selection metric.

**Tech Stack:** Python 3.11, PyTorch, timm, torchvision, pandas, PIL, YAML configs, unittest

---

## File Map

### Create

- `docs/superpowers/handoffs/EXTERNAL_SIGNAL_DATASET_AUDIT.md`
- `docs/superpowers/handoffs/EXTERNAL_SIGNAL_PLAN.md`
- `docs/superpowers/handoffs/EXTERNAL_SIGNAL_RESULTS.md`
- `src/external_data/slid.py`
- `src/pretraining/train_slid_cornea_pretrain.py`
- `configs/pretrain_slid_convnextv2_tiny_cornea_mask.yaml`
- `configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__currentbranch_control.yaml`
- `configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__from_slid_cornea_pretrain.yaml`
- `tests/test_external_signal_configs.py`

### Modify

- `src/model_factory.py`
- `src/main_train.py`
- `README_training.md`
- `codex.md`

### Outputs Expected From Execution

- `models/checkpoints/pretrain__slid__convnextv2_tiny__cornea_mask__seed42/best.pt`
- `models/exported/pretrain__slid__convnextv2_tiny__cornea_mask__seed42/best.pt`
- `models/checkpoints/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42__currentbranch_control/best.pt`
- `models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42__currentbranch_control/best.pt`
- `models/checkpoints/pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt`
- `models/exported/pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt`
- `outputs/metrics/<experiment>/{val,test}_metrics.json`
- `outputs/reports/<experiment>/{val,test}_summary.md`
- `outputs/predictions/<experiment>/{val,test}_predictions.csv`

### Runtime Interpreter

Use the direct environment interpreter:

- `/home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python`

All commands below assume:

```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python ...
```

## Task 1: Audit External Dataset Reality

**Files:**
- Create: `docs/superpowers/handoffs/EXTERNAL_SIGNAL_DATASET_AUDIT.md`

- [ ] Inspect `data/external/slid`, `data/interim/slid`, and any prior local `SLID` manifests or split files.
- [ ] Record, for each reused `SLID` artifact, exact path, sample count, expected schema/file layout, and a quick integrity result.
- [ ] Verify whether current-branch code can consume the reused `SLID` artifacts directly or whether a thin adapter is required.
- [ ] Audit `SLIT-Net` access from public sources and state clearly whether it is actually downloadable and usable in practice.
- [ ] Document modality, label types, dataset size, structure, and recommended repo role for both datasets.
- [ ] Stop `SLIT-Net` from affecting the base run if access, terms, or structure remain unclear.

Completion rule:
- `EXTERNAL_SIGNAL_DATASET_AUDIT.md` exists and clearly names what is reused, what is not reused, and why.

## Task 2: Lock The Executed External-Signal Plan

**Files:**
- Create: `docs/superpowers/handoffs/EXTERNAL_SIGNAL_PLAN.md`

- [ ] State the base upstream task explicitly as `SLID` cornea-mask supervision.
- [ ] State the default initialization chain explicitly:
  - ImageNet-pretrained `convnextv2_tiny`
  - `SLID` cornea-mask external pretraining
  - SUSTech `pattern_3class` fine-tune
- [ ] State the downstream invariants:
  - `pattern_3class`
  - `cornea_crop_scale_v1`
  - `pattern_augplus_v2`
  - `weighted_sampler_tempered`
  - validation `balanced_accuracy`
- [ ] State the mandatory executed rows:
  - frozen official benchmark
  - frozen deployed late-fusion rule
  - current-branch control
  - current-branch `SLID` warm-start
- [ ] State the fifth-row gate:
  - only after the 4-row comparison executes cleanly
  - only if justified
  - only if it does not delay or bloat the base run

Completion rule:
- `EXTERNAL_SIGNAL_PLAN.md` matches the approved spec and freezes the base run before implementation grows.

## Task 3: Add Minimal SLID Dataset Integration

**Files:**
- Create: `src/external_data/slid.py`
- Create: `tests/test_external_signal_configs.py`

- [ ] Write a thin `SLID` dataset adapter around the audited manifest/layout rather than building a general framework.
- [ ] Support the verified base-task payload needed for cornea-mask supervision.
- [ ] Add a small test that verifies the expected config / adapter wiring resolves without using SUSTech labels.
- [ ] Keep `SLID` integration separate from the core SUSTech `data/dataset.py` path unless a very small shared helper is obviously cleaner.

Completion rule:
- current-branch code can load audited `SLID` samples for the chosen upstream task without touching archived task families.

## Task 4: Add External Pretraining Entrypoint

**Files:**
- Create: `src/pretraining/train_slid_cornea_pretrain.py`
- Create: `configs/pretrain_slid_convnextv2_tiny_cornea_mask.yaml`
- Modify: `src/model_factory.py`

- [ ] Add one external pretraining entrypoint for `SLID` cornea-mask supervision.
- [ ] Implement the base upstream task explicitly as binary cornea segmentation from audited `SLID` cornea masks, not as an ambiguous generic auxiliary objective.
- [ ] Reuse the existing ConvNeXtV2 / `timm` model construction path where practical.
- [ ] Ensure the exported checkpoint name identifies both dataset and upstream task:
  - `pretrain__slid__convnextv2_tiny__cornea_mask__seed42`
- [ ] Export a checkpoint whose lineage is suitable for downstream warm-start.
- [ ] Keep the pretraining objective single-task by default.

Completion rule:
- a current-branch external checkpoint is produced and named with explicit dataset/task lineage.

## Task 5: Add Warm-Start Hook To Canonical Pattern Training

**Files:**
- Modify: `src/main_train.py`
- Create: `configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__currentbranch_control.yaml`
- Create: `configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__from_slid_cornea_pretrain.yaml`
- Modify: `tests/test_external_signal_configs.py`

- [ ] Add one explicit warm-start config field, e.g. `warmstart_checkpoint`.
- [ ] Load only the backbone weights by default.
- [ ] Add a hard warm-start compatibility check that fails clearly if the upstream checkpoint backbone shape does not match the downstream `convnextv2_tiny` backbone.
- [ ] Do not silently partially load mismatched backbone weights unless that behavior is explicitly implemented and logged.
- [ ] Reinitialize the downstream `pattern_3class` classifier head explicitly.
- [ ] Add one current-branch control config with no external warm-start.
- [ ] Add one downstream warm-start config whose name identifies external slit-lamp pretraining:
  - `pattern3__convnextv2_tiny__external_slitlamp_pretrain__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
- [ ] Add a config test that verifies the control and warm-start configs remain identical except for experiment naming and warm-start lineage fields.

Completion rule:
- the canonical downstream pattern path can run both control and warm-start variants from current mainline code.

## Task 6: Execute The Current-Branch Control

**Files:**
- Output only

- [ ] Train the current-branch canonical no-warm-start control.
- [ ] Use the explicit `__currentbranch_control` config for the control run rather than substituting an older equivalent config path.
- [ ] Run `val` and `test` evaluation through `src/main_eval.py`.
- [ ] Record exact commands and artifact paths for the results doc.
- [ ] If local drift makes the control clearly inconsistent or unreproducible, record that limitation and continue with frozen-baseline comparison honesty intact.

Completion rule:
- control artifacts exist, or the exact limitation is documented with enough detail to support the stop rule.

## Task 7: Execute The SLID Warm-Start Experiment

**Files:**
- Output only

- [ ] Train the `SLID` cornea-mask external stage from the current branch.
- [ ] Fine-tune the downstream pattern model from that external checkpoint using the fixed pattern recipe.
- [ ] Run `val` and `test` evaluation through `src/main_eval.py`.
- [ ] Record exact pretrain checkpoint path and exact downstream warm-start source used.
- [ ] Persist warm-start lineage automatically into downstream run metadata and reports so the exact upstream checkpoint path is recorded with the executed experiment.

Completion rule:
- one full current-branch external-signal experiment exists end to end with explicit checkpoint lineage.

## Task 8: Run Leakage And Contamination Checks

**Files:**
- `docs/superpowers/handoffs/EXTERNAL_SIGNAL_RESULTS.md`

- [ ] Verify canonical SUSTech split preservation.
- [ ] Verify no change to `pattern_3class` label definitions.
- [ ] Verify no external-data relabeling of SUSTech targets.
- [ ] Verify no val/test-driven preprocessing or model selection.
- [ ] Check filename/path overlaps between external data and SUSTech data where practical.
- [ ] Check duplicate IDs or manifest collisions across datasets where practical.
- [ ] Run hash-based or exact-file duplicate checks where practical and document what was checked.
- [ ] Check for obvious overlap or contamination signals between external data and SUSTech evaluation data.
- [ ] If overlap is unclear and material, stop and document it plainly.

Completion rule:
- the results doc contains an explicit leakage/contamination section, not just an implicit assumption.

## Task 9: Write Final Results Handoff

**Files:**
- Create: `docs/superpowers/handoffs/EXTERNAL_SIGNAL_RESULTS.md`
- Modify if justified: `README_training.md`
- Modify if justified: `codex.md`

- [ ] Report the required rows:
  - frozen official single-model benchmark
  - frozen deployed late-fusion rule
  - current-branch control
  - current-branch `SLID` warm-start
  - optional fifth row only if the base 4-row comparison already finished cleanly
- [ ] Report val/test balanced accuracy and macro F1.
- [ ] Report delta vs the frozen official benchmark and the frozen deployed rule.
- [ ] Report delta vs current-branch control so the effect of `SLID` in the current branch environment is visible.
- [ ] Report whether the result is promising, strong, or not worth keeping using the predeclared bar.
- [ ] If the warm-start loses, say so plainly.
- [ ] If old checkpoints are mentioned, mark them historical-only.
- [ ] Include exact commands run, exact artifact paths, exact external checkpoint path, and exact warm-start source.

Completion rule:
- `EXTERNAL_SIGNAL_RESULTS.md` is sufficient for a third party to see what was actually executed and whether the new signal helped.

## Optional Task 10: Fifth Row Only If Earned

**Files:**
- as justified by execution

- [ ] Confirm the base 4-row comparison completed cleanly first.
- [ ] Confirm the extra run will not delay or bloat the core experiment.
- [ ] Choose only one:
  - `SLIT-Net`-based line if the audit proved it is clean and usable
  - one second `SLID` variant if needed for a fair comparison

Completion rule:
- skip this task unless the gate is satisfied.

## Final Acceptance Checklist

- [ ] `SLID` and `SLIT-Net` accessibility were actually audited
- [ ] reused local `SLID` artifacts were documented and verified
- [ ] one current-branch external pretrain checkpoint was produced
- [ ] one current-branch canonical control was executed if possible
- [ ] one current-branch `SLID` warm-start pattern run was executed
- [ ] exact val/test metrics were reported
- [ ] exact checkpoint lineage was reported
- [ ] frozen truth stayed frozen
- [ ] scope stayed pattern-only
