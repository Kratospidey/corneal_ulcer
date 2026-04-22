# WORKING SEV-S1 Geometry Note

## Branch State

- Clean worktree: `/tmp/corneal_ulcer_sev_s1_audit`
- Branch: `exp/sev-s1-geometry-audit`
- Base: `exp/tg-severity-rescue`
- Pattern line remains frozen.

## Environment

- `micromamba run -n corneal-train python -V` -> `Python 3.11.15`
- `micromamba run -n corneal-train python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"` -> `2.10.0+cu128`, `True`
- CUDA available on local RTX 5070
- `xgboost` unavailable in env

## Files Added

- `src/experimental/severity/cornea_circle_refine.py`
- `src/experimental/severity/build_geometry_table.py`
- `src/experimental/severity/train_tabular.py`
- `src/experimental/severity/eval_severity_posthoc.py`
- `src/experimental/tg/run_punctate_audit.py`
- `docs/superpowers/handoffs/TG_PUNCTATE_AUDIT.md`
- `docs/superpowers/handoffs/SEV_S1_RESULTS.md`
- `docs/superpowers/handoffs/WORKING_SEV_S1_GEOMETRY_NOTE.md`

## Commands Run

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/tg/run_punctate_audit.py \
  --manifest /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/manifests/manifest.csv \
  --split-file /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/split_files/task_tg_5class_holdout.csv \
  --tg-experiment tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42 \
  --artifact-root outputs \
  --output-dir outputs/debug/tg_punctate_audit \
  --report-path docs/superpowers/handoffs/TG_PUNCTATE_AUDIT.md \
  --repo-root /tmp/corneal_ulcer_sev_s1_audit
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/build_geometry_table.py \
  --manifest /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/manifests/manifest.csv \
  --split-file /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/split_files/pattern_3class_holdout.csv \
  --output-path outputs/debug/severity_posthoc/geom_table_geomonly_v1.csv
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/build_geometry_table.py \
  --manifest /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/manifests/manifest.csv \
  --split-file /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/split_files/pattern_3class_holdout.csv \
  --output-path outputs/debug/severity_posthoc/geom_table_v1.csv \
  --pattern-config /home/kratospidey/Repos/corneal_ulcer_classification/configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml \
  --pattern-checkpoint /home/kratospidey/Repos/corneal_ulcer_classification/models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt \
  --pattern-device cuda
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_tabular.py \
  --table outputs/debug/severity_posthoc/geom_table_geomonly_v1.csv \
  --model hgb \
  --experiment-name severity5__posthoc__geom_hgb_v1__holdout_v1 \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_tabular.py \
  --table outputs/debug/severity_posthoc/geom_table_v1.csv \
  --model hgb \
  --experiment-name severity5__posthoc__geom_plus_patternlogits_hgb_v1__holdout_v1 \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_tabular.py \
  --table outputs/debug/severity_posthoc/geom_table_v1.csv \
  --model rules \
  --experiment-name severity5__posthoc__geomrules_v1__holdout_v1 \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/eval_severity_posthoc.py \
  --experiment-name severity5__posthoc__geomrules_v1__holdout_v1 \
  --experiment-name severity5__posthoc__geom_hgb_v1__holdout_v1 \
  --experiment-name severity5__posthoc__geom_plus_patternlogits_hgb_v1__holdout_v1 \
  --output-root outputs \
  --report-path docs/superpowers/handoffs/SEV_S1_RESULTS.md
```

## TG Audit Outcome

- Exact A3 seed-42 rerun reproduced the frozen negative result.
- `type3` test count is `1`; it was predicted as `patch_gt_1mm` with confidence `0.980`.
- `macro_punctate` and `coalescent_macro_punctate` both have zero recall on test.
- No conflicting punctate duplicate-label groups were found in `duplicate_candidates.csv`.

## Severity Outcome

- Geometry-only HGB is effectively tied with the old fallback.
- Geometry plus frozen pattern logits HGB is the best branch result:
  - BA `0.3993`
  - macro F1 `0.4020`
  - central recall `0.6923`
- Still materially below the strict learned severity reference.

## Current Recommendation

- Continue SEV-S1 as a post-hoc branch only if the next step is still disciplined.
- Do not promote TG from this line.
- If TG is revisited later, do it through a targeted multiscale diagnostic transplant, not another loss tweak pass.
