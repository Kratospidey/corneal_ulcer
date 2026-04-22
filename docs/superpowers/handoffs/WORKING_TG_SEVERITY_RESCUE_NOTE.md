# Working TG + Severity Rescue Note

## 1. Freeze-doc audit

- Freeze branch: `freeze/official-handoff-2026-04-21`
- Clean continuation branch: `exp/tg-severity-rescue`
- Official frozen single model remains:
  - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - test balanced accuracy `0.8481921571`
  - test macro F1 `0.7989648033`
- Best deployed inference rule remains separate:
  - `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
  - test balanced accuracy `0.8563222384`
  - test macro F1 `0.8114845938`
- Pattern stays frozen.
- TG remains the only learned continuation worth touching.
- Severity remains post-hoc / geometry-aware / experimental.
- Runbook warning is real:
  - latest TG rescue implementation is not committed on the freeze branch
  - latest severity salvage implementation is not committed on the freeze branch

## 2. Existing scripts and configs

### Committed on the freeze branch

- `src/main_train.py`
- `src/main_eval.py`
- `src/model_factory.py`
- `src/training/losses.py`
- `src/data/transforms.py`
- `configs/splits.yaml`
- baseline pattern and severity configs only

### Present only in the dirty research tree, not committed on the freeze branch

- `src/main_train_hierarchical.py`
- `src/main_train_severity_structured.py`
- `src/main_feature_baseline.py`
- `src/mask_polarity_debug.py`
- `src/run_ulcer_supervision_audit.py`
- multiple severity and pattern+severity configs

### Runtime artifacts available only in the original repo tree

- manifest: `/home/kratospidey/Repos/corneal_ulcer_classification/data/interim/manifests/manifest.csv`
- TG split file: `/home/kratospidey/Repos/corneal_ulcer_classification/data/interim/split_files/task_tg_5class_holdout.csv`
- duplicate candidates: `/home/kratospidey/Repos/corneal_ulcer_classification/outputs/tables/duplicate_candidates.csv`
- official warm-start checkpoint:
  `/home/kratospidey/Repos/corneal_ulcer_classification/models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt`

## 3. Gaps that required new code/config

- Freeze branch did not have a committed TG task config.
- Freeze branch did not have augmentation-profile support for the official `augplus_v2` family.
- Freeze branch did not have warm-start backbone loading.
- Freeze branch had no structured TG `T1/T2/T3` loss.
- Freeze branch had no TG-focused evaluation utility for punctate-family metrics and `type3` / `type4` guardrails.
- Clean worktree is missing gitignored runtime assets, so configs must point at existing local artifacts in the original repo tree.

## 4. New local continuation files added on `exp/tg-severity-rescue`

- `configs/task_tg_5class.yaml`
- `configs/splits_runtime_mainrepo.yaml`
- `configs/train_tg_structured_t123_balsoftmax_t3.yaml`
- `src/run_tg_eval_summary.py`

### Minimal baseline patches

- `src/data/transforms.py`
- `src/data/label_utils.py`
- `src/model_factory.py`
- `src/training/losses.py`
- `src/main_train.py`
- `src/main_eval.py`

## 5. Planned experiment names

- `tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42`
- `tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed73`
  only if `seed42` earns it
- `tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed17`
  only if `seed42` earns it
- `tg5__convnextv2_tiny__officialwarm__structured_t123__ldam_drw_t3__holdout_v1__seed42`
  only if BalSoftmax underdelivers
- `tg5__convnextv2_tiny__officialwarm__structured_t123__t3_multiscale_v1__holdout_v1__seed42`
  only if TG-A3 shows real signal
- `severity5__posthoc__officialpattern_embed32_plus_geom_v1__rules__holdout_v1`
- `severity5__posthoc__officialpattern_embed32_plus_geom_v1__logreg__holdout_v1`
- `severity5__posthoc__officialpattern_embed32_plus_geom_v1__hgb__holdout_v1`

## 6. Exact commands planned now

### Runtime asset bridge

```bash
mkdir -p data/raw
ln -s /home/kratospidey/Repos/corneal_ulcer_classification/data/raw/sustech_sysu data/raw/sustech_sysu
```

### TG-A3 seed42 training

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/main_train.py \
  --config configs/train_tg_structured_t123_balsoftmax_t3.yaml \
  --device cuda \
  --seed-override 42 \
  --experiment-name-override tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42
```

### TG-A3 seed42 evaluation

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/main_eval.py \
  --config configs/train_tg_structured_t123_balsoftmax_t3.yaml \
  --checkpoint models/checkpoints/tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42/best.pt \
  --split test \
  --device cuda \
  --experiment-name-override tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42
```

### TG-focused metrics export

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/run_tg_eval_summary.py \
  --experiment-name tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42 \
  --split test
```

### Severity preflight checks, only after TG-A3

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/mask_polarity_debug.py
PYTHONPATH=src micromamba run -n corneal-train python src/run_ulcer_supervision_audit.py
```

## 7. Current hard blockers still in scope

- The freeze docs reference `outputs/debug/2026-04-21_post_unified_rescue_pass/...`, but those evidence files are missing from the available workspace.
- The clean worktree does not contain gitignored runtime data or checkpoints; it must reuse the local artifacts already present in the original repo tree.
- There is still no committed SEV-S1 post-hoc geometry pipeline. That branch will need a separate feasibility check after TG-A3.
