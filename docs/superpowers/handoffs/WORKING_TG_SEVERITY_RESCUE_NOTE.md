# Working TG + Severity Rescue Note

Historical note:

- This was a transient working note from before the final stop decision.
- It is now superseded by:
  - `FINAL_TG_STATUS.md`
  - `FINAL_SEVERITY_STATUS.md`
  - `FINAL_PATTERN_ONLY_DECISION.md`

Current truth:

- Pattern only
- TG archived
- Severity archived

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
