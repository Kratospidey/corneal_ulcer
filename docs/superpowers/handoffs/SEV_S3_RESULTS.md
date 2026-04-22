# SEV-S3 Extent-Focused S2 Rescue

## Scope

- Pattern stayed frozen.
- TG stayed paused.
- Severity remained fully post-hoc.
- S0 and S1 were kept fixed from the best SEV-S2 scaffold.
- Only the noncentral S2 extent branch was changed.

## Frozen Truth Restated

- Official frozen single model:
  - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - test balanced accuracy `0.8482`
  - test macro F1 `0.7990`
- Best deployed inference rule:
  - `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
  - test balanced accuracy `0.8563`
  - test macro F1 `0.8115`
- Best SEV-S1:
  - `severity5__posthoc__geom_plus_patternlogits_hgb_v1__holdout_v1`
  - balanced accuracy `0.3993`
  - macro F1 `0.4020`
- Best SEV-S2:
  - `severity5__posthoc__factorized_geom_plus_patternlogits_hgb_v1__holdout_v1`
  - balanced accuracy `0.4213`
  - macro F1 `0.4347`
- TG remains paused because the punctate audit concluded `data scarcity + hierarchy starvation`, not a small-loss-tweak problem.

## Files Added Or Modified

- `src/experimental/severity/build_geometry_table.py`
- `src/experimental/severity/build_s2_ordinal_tables.py`
- `src/experimental/severity/train_s2_ordinal_tabular.py`
- `src/experimental/severity/eval_s2_ordinal_severity.py`
- `docs/superpowers/handoffs/SEV_S3_RESULTS.md`
- `docs/superpowers/handoffs/WORKING_SEV_S3_NOTE.md`

## Commands Run

```bash
micromamba run -n corneal-train python -V
micromamba run -n corneal-train python -c "import torch, sklearn, importlib.util; print(torch.__version__); print(torch.cuda.is_available()); print(sklearn.__version__); print(importlib.util.find_spec('xgboost') is not None)"
python -m py_compile src/experimental/severity/build_geometry_table.py src/experimental/severity/build_s2_ordinal_tables.py src/experimental/severity/train_s2_ordinal_tabular.py src/experimental/severity/eval_s2_ordinal_severity.py
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/build_geometry_table.py \
  --manifest /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/manifests/manifest.csv \
  --split-file /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/split_files/pattern_3class_holdout.csv \
  --output-path outputs/debug/severity_posthoc/geom_table_v3.csv \
  --pattern-config /home/kratospidey/Repos/corneal_ulcer_classification/configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml \
  --pattern-checkpoint /home/kratospidey/Repos/corneal_ulcer_classification/models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt \
  --pattern-device cuda
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/build_s2_ordinal_tables.py \
  --table outputs/debug/severity_posthoc/geom_table_v3.csv \
  --output-dir outputs/debug/severity_posthoc/s2_ordinal_v1
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_s2_ordinal_tabular.py \
  --table outputs/debug/severity_posthoc/s2_ordinal_v1/s2_flat.csv \
  --model rules \
  --experiment-name severity5__posthoc__factorized_s2_flat_rules_v1__holdout_v1 \
  --feature-mode geom_only \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_s2_ordinal_tabular.py \
  --table outputs/debug/severity_posthoc/s2_ordinal_v1/s2_flat.csv \
  --model hgb \
  --experiment-name severity5__posthoc__factorized_s2_flat_hgb_v2__holdout_v1 \
  --feature-mode all_numeric \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_s2_ordinal_tabular.py \
  --table outputs/debug/severity_posthoc/s2_ordinal_v1/s2_leq25.csv \
  --model rules \
  --experiment-name severity5__posthoc__factorized_s2_leq25_rules_v1__holdout_v1 \
  --feature-mode geom_only \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_s2_ordinal_tabular.py \
  --table outputs/debug/severity_posthoc/s2_ordinal_v1/s2_leq25.csv \
  --model hgb \
  --experiment-name severity5__posthoc__factorized_s2_leq25_hgb_v1__holdout_v1 \
  --feature-mode all_numeric \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_s2_ordinal_tabular.py \
  --table outputs/debug/severity_posthoc/s2_ordinal_v1/s2_geq75.csv \
  --model rules \
  --experiment-name severity5__posthoc__factorized_s2_geq75_rules_v1__holdout_v1 \
  --feature-mode geom_only \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_s2_ordinal_tabular.py \
  --table outputs/debug/severity_posthoc/s2_ordinal_v1/s2_geq75.csv \
  --model hgb \
  --experiment-name severity5__posthoc__factorized_s2_geq75_hgb_v1__holdout_v1 \
  --feature-mode all_numeric \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/eval_s2_ordinal_severity.py \
  --geom-table outputs/debug/severity_posthoc/geom_table_v3.csv \
  --s0-experiment severity5__posthoc__factorized_s0_plus_patternlogits_hgb_v1__holdout_v1 \
  --s1-experiment severity5__posthoc__factorized_s1_plus_patternlogits_hgb_v1__holdout_v1 \
  --s2-flat-experiment severity5__posthoc__factorized_s2_flat_hgb_v2__holdout_v1 \
  --s2-leq25-experiment severity5__posthoc__factorized_s2_leq25_hgb_v1__holdout_v1 \
  --s2-geq75-experiment severity5__posthoc__factorized_s2_geq75_hgb_v1__holdout_v1 \
  --flat-experiment-name severity5__posthoc__factorized_geom_plus_patternlogits_hgb_v2__holdout_v1 \
  --ordinal-experiment-name severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1 \
  --output-root outputs \
  --report-path docs/superpowers/handoffs/SEV_S3_RESULTS.md
```

## SEV-S3 Feature Extension

Added narrow S2-focused features to `geom_table_v3.csv`:

- `response_area_frac_t3_0`
- `response_area_frac_t3_5`
- `lesion_mass_top25_fraction`
- `lesion_mass_top10_fraction`
- `paracentral_minus_peripheral_fraction`

These were added on top of the SEV-S2 geometry table, which already carried:

- multi-threshold lesion area fractions
- ring fractions and central/peripheral occupancy
- lesion centroid and min-distance-to-center
- connected-component count and largest-component fraction
- green-response summary statistics
- simple pattern-logit interaction features

## S2 Routing

- Source S2 rows: `574`
- Split counts:
  - train `399`
  - val `87`
  - test `88`
- Flat S2 counts:
  - `ulcer_leq_25pct=98`
  - `ulcer_leq_50pct=203`
  - `ulcer_geq_75pct=273`
- Threshold B counts:
  - `ulcer_leq_25pct=98`
  - `greater_than_25pct=476`
- Threshold C counts:
  - `less_than_75pct=301`
  - `ulcer_geq_75pct=273`

## S2 Stage Metrics

| Stage Experiment | BA | Macro F1 | Key Note |
| --- | ---: | ---: | --- |
| `severity5__posthoc__factorized_s2_flat_rules_v1__holdout_v1` | 0.2806 | 0.2153 | `ulcer_leq_50pct` collapsed entirely |
| `severity5__posthoc__factorized_s2_flat_hgb_v2__holdout_v1` | 0.4537 | 0.4536 | Best flat S2 stage in this branch |
| `severity5__posthoc__factorized_s2_leq25_rules_v1__holdout_v1` | 0.4447 | 0.3596 | High `<=25` recall but too many false positives |
| `severity5__posthoc__factorized_s2_leq25_hgb_v1__holdout_v1` | 0.5776 | 0.5711 | Better `<=25` gate than flat stage |
| `severity5__posthoc__factorized_s2_geq75_rules_v1__holdout_v1` | 0.4760 | 0.4748 | Weak high-extent separation |
| `severity5__posthoc__factorized_s2_geq75_hgb_v1__holdout_v1` | 0.6245 | 0.6246 | Strongest individual threshold stage |

## Combined System Metrics

Fixed upstream stages:

- S0: `severity5__posthoc__factorized_s0_plus_patternlogits_hgb_v1__holdout_v1`
- S1: `severity5__posthoc__factorized_s1_plus_patternlogits_hgb_v1__holdout_v1`

Combined comparison:

| Combined Experiment | BA | Macro F1 | No-Ulcer Precision | Central Recall | leq25 Recall | leq25 F1 | Adjacent Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `severity5__posthoc__factorized_geom_plus_patternlogits_hgb_v2__holdout_v1` | 0.4213 | 0.4321 | 1.0000 | 0.6923 | 0.1333 | 0.1429 | 0.6481 |
| `severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1` | 0.4233 | 0.4366 | 1.0000 | 0.6923 | 0.2000 | 0.1765 | 0.6727 |

## Exact Comparison Against References

Frozen references:

- `hgb_fallback`: BA `0.3280`, macro F1 `0.3327`
- best SEV-S1: BA `0.3993`, macro F1 `0.4020`
- best SEV-S2: BA `0.4213`, macro F1 `0.4347`
- strict severity reference: BA `0.6109`, macro F1 `0.5542`

Ordinal SEV-S3 deltas:

- vs `hgb_fallback`:
  - `+0.0953` BA
  - `+0.1038` macro F1
- vs best SEV-S1:
  - `+0.0240` BA
  - `+0.0346` macro F1
- vs best SEV-S2:
  - `+0.0020` BA
  - `+0.0018` macro F1
- vs strict reference:
  - `-0.1876` BA
  - `-0.1177` macro F1

## Mild-Bin Diagnostics

For `severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1`:

- `ulcer_leq_25pct` prediction breakdown:
  - `ulcer_leq_25pct=3`
  - `ulcer_leq_50pct=6`
  - `ulcer_geq_75pct=5`
  - `central_ulcer=1`
- `ulcer_leq_50pct` prediction breakdown:
  - `ulcer_leq_25pct=4`
  - `ulcer_leq_50pct=15`
  - `ulcer_geq_75pct=10`
  - `central_ulcer=1`
- no-ulcer false negatives: `6`
- no-ulcer false positives: `0`
- ordinal threshold conflict count: `3`
- conflict resolution rule: `>=75` wins

Interpretation:

- The ordinal S2 rescue did improve the mildest class:
  - `ulcer_leq_25pct` recall improved from `0.1333` to `0.2000`
  - `ulcer_leq_25pct` F1 improved from `0.1429` to `0.1765`
- It preserved the useful SEV-S2 behavior:
  - no-ulcer precision stayed `1.0000`
  - central-ulcer recall stayed `0.6923`
- The gain is still narrow:
  - BA only improved by `0.0020`
  - macro F1 only improved by `0.0018`
  - adjacent-class error worsened from `0.5741` in best SEV-S2 to `0.6727`

## Bugs Found And Fixed

- The ordinal evaluator had dead diagnostics code that could fail at runtime; removed.
- The S2 routed tables initially omitted `factorized_route`, which broke the reused prediction writer; added.
- The evaluator initially recomputed feature columns from `geom_table_v3.csv` and handed S0/S1 HGB extra features; changed it to respect each stage model’s saved training columns.

## Blunt Verdict

- Branch verdict: `useful but incomplete`

Why:

- SEV-S3 does beat the best SEV-S2 run, but only narrowly.
- The main win is localized:
  - slightly better `ulcer_leq_25pct` handling
  - preserved `1.0` no-ulcer precision
- The core noncentral extent problem is still unresolved.
- This is not strong enough to claim a clean rescue of severity.
