# SEV-S2 Factorized Post-Hoc Results

## Scope

- Pattern stayed frozen.
- TG stayed paused.
- Severity remained fully post-hoc.
- No shared-backbone severity head was added.
- Historical ulcer-mask absence was not used as supervision.

## Files Added Or Modified

- `src/experimental/severity/build_geometry_table.py`
- `src/experimental/severity/build_factorized_tables.py`
- `src/experimental/severity/train_factorized_tabular.py`
- `src/experimental/severity/eval_factorized_severity.py`
- `docs/superpowers/handoffs/SEV_S2_RESULTS.md`
- `docs/superpowers/handoffs/WORKING_SEV_S2_NOTE.md`

## Commands Run

```bash
git checkout -b exp/sev-s2-factorized-posthoc
```

```bash
micromamba run -n corneal-train python -V
micromamba run -n corneal-train python -c "import torch, sklearn, importlib.util; print(torch.__version__); print(torch.cuda.is_available()); print(sklearn.__version__); print(importlib.util.find_spec('xgboost') is not None)"
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/build_geometry_table.py \
  --manifest /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/manifests/manifest.csv \
  --split-file /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/split_files/pattern_3class_holdout.csv \
  --output-path outputs/debug/severity_posthoc/geom_table_v2.csv \
  --pattern-config /home/kratospidey/Repos/corneal_ulcer_classification/configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml \
  --pattern-checkpoint /home/kratospidey/Repos/corneal_ulcer_classification/models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt \
  --pattern-device cuda
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/build_factorized_tables.py \
  --table outputs/debug/severity_posthoc/geom_table_v2.csv \
  --output-dir outputs/debug/severity_posthoc/factorized_v1
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_factorized_tabular.py \
  --table outputs/debug/severity_posthoc/factorized_v1/s0_table.csv \
  --model rules \
  --experiment-name severity5__posthoc__factorized_s0_rules_v1__holdout_v1 \
  --feature-mode geom_only \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_factorized_tabular.py \
  --table outputs/debug/severity_posthoc/factorized_v1/s1_table.csv \
  --model rules \
  --experiment-name severity5__posthoc__factorized_s1_rules_v1__holdout_v1 \
  --feature-mode geom_only \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_factorized_tabular.py \
  --table outputs/debug/severity_posthoc/factorized_v1/s2_table.csv \
  --model rules \
  --experiment-name severity5__posthoc__factorized_s2_rules_v1__holdout_v1 \
  --feature-mode geom_only \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_factorized_tabular.py \
  --table outputs/debug/severity_posthoc/factorized_v1/s0_table.csv \
  --model hgb \
  --experiment-name severity5__posthoc__factorized_s0_hgb_v1__holdout_v1 \
  --feature-mode geom_only \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_factorized_tabular.py \
  --table outputs/debug/severity_posthoc/factorized_v1/s1_table.csv \
  --model hgb \
  --experiment-name severity5__posthoc__factorized_s1_hgb_v1__holdout_v1 \
  --feature-mode geom_only \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_factorized_tabular.py \
  --table outputs/debug/severity_posthoc/factorized_v1/s2_table.csv \
  --model hgb \
  --experiment-name severity5__posthoc__factorized_s2_hgb_v1__holdout_v1 \
  --feature-mode geom_only \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_factorized_tabular.py \
  --table outputs/debug/severity_posthoc/factorized_v1/s0_table.csv \
  --model hgb \
  --experiment-name severity5__posthoc__factorized_s0_plus_patternlogits_hgb_v1__holdout_v1 \
  --feature-mode all_numeric \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_factorized_tabular.py \
  --table outputs/debug/severity_posthoc/factorized_v1/s1_table.csv \
  --model hgb \
  --experiment-name severity5__posthoc__factorized_s1_plus_patternlogits_hgb_v1__holdout_v1 \
  --feature-mode all_numeric \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/train_factorized_tabular.py \
  --table outputs/debug/severity_posthoc/factorized_v1/s2_table.csv \
  --model hgb \
  --experiment-name severity5__posthoc__factorized_s2_plus_patternlogits_hgb_v1__holdout_v1 \
  --feature-mode all_numeric \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/eval_factorized_severity.py \
  --geom-table outputs/debug/severity_posthoc/geom_table_v2.csv \
  --s0-experiment severity5__posthoc__factorized_s0_rules_v1__holdout_v1 \
  --s1-experiment severity5__posthoc__factorized_s1_rules_v1__holdout_v1 \
  --s2-experiment severity5__posthoc__factorized_s2_rules_v1__holdout_v1 \
  --experiment-name severity5__posthoc__factorized_geomrules_v1__holdout_v1 \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/eval_factorized_severity.py \
  --geom-table outputs/debug/severity_posthoc/geom_table_v2.csv \
  --s0-experiment severity5__posthoc__factorized_s0_hgb_v1__holdout_v1 \
  --s1-experiment severity5__posthoc__factorized_s1_hgb_v1__holdout_v1 \
  --s2-experiment severity5__posthoc__factorized_s2_hgb_v1__holdout_v1 \
  --experiment-name severity5__posthoc__factorized_geom_hgb_v1__holdout_v1 \
  --output-root outputs
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/eval_factorized_severity.py \
  --geom-table outputs/debug/severity_posthoc/geom_table_v2.csv \
  --s0-experiment severity5__posthoc__factorized_s0_plus_patternlogits_hgb_v1__holdout_v1 \
  --s1-experiment severity5__posthoc__factorized_s1_plus_patternlogits_hgb_v1__holdout_v1 \
  --s2-experiment severity5__posthoc__factorized_s2_plus_patternlogits_hgb_v1__holdout_v1 \
  --experiment-name severity5__posthoc__factorized_geom_plus_patternlogits_hgb_v1__holdout_v1 \
  --output-root outputs
```

## Geometry Table Extension

Added features in `geom_table_v2.csv` beyond SEV-S1:

- extra response thresholds: `response_area_frac_t2_0`, `response_area_frac_t2_5`
- extra weighted response features: `response_weighted_area_t1_0`, `response_weighted_area_t1_5`, `central_response_weighted_area`
- extra response statistics: `green_response_std`, `green_response_z_mean`, `green_response_z_std`, `green_response_z_q95`
- extra interpretable geometry interactions: `response_area_ratio_t1_5_to_t0_5`, `central_minus_peripheral_fraction`
- optional pattern interactions: `pattern_flaky_x_response_area_t1_0`, `pattern_flaky_x_component_count`, `pattern_mixed_x_central_occupancy`, `pattern_confidence_x_response_weighted_area`

## Factorized Routing

- S0:
  - `no_ulcer`
  - `ulcer_present`
- S1 among `ulcer_present`:
  - `central_ulcer`
  - `noncentral_ulcer`
- S2 among `noncentral_ulcer`:
  - `ulcer_leq_25pct`
  - `ulcer_leq_50pct`
  - `ulcer_geq_75pct`

Counts:

- S0: `no_ulcer=36`, `ulcer_present=676`
- S1: `central_ulcer=102`, `noncentral_ulcer=574`
- S2: `ulcer_leq_25pct=98`, `ulcer_leq_50pct=203`, `ulcer_geq_75pct=273`

## Stage-Wise Test Metrics

| Stage Model | BA | Macro F1 | Note |
| --- | ---: | ---: | --- |
| `severity5__posthoc__factorized_s0_rules_v1__holdout_v1` | 0.6818 | 0.5598 | High `no_ulcer` recall, weak precision |
| `severity5__posthoc__factorized_s1_rules_v1__holdout_v1` | 0.5476 | 0.4613 | Too many noncentral false routes |
| `severity5__posthoc__factorized_s2_rules_v1__holdout_v1` | 0.3672 | 0.3202 | Extent stage too crude |
| `severity5__posthoc__factorized_s0_hgb_v1__holdout_v1` | 0.6280 | 0.6471 | Better macro balance than rules |
| `severity5__posthoc__factorized_s1_hgb_v1__holdout_v1` | 0.6368 | 0.6670 | Better than S1 rules, still misses central cases |
| `severity5__posthoc__factorized_s2_hgb_v1__holdout_v1` | 0.4693 | 0.4726 | Best geom-only extent stage |
| `severity5__posthoc__factorized_s0_plus_patternlogits_hgb_v1__holdout_v1` | 0.5714 | 0.6106 | No-ulcer recall drops hard |
| `severity5__posthoc__factorized_s1_plus_patternlogits_hgb_v1__holdout_v1` | 0.8234 | 0.8234 | Strong central vs noncentral gate |
| `severity5__posthoc__factorized_s2_plus_patternlogits_hgb_v1__holdout_v1` | 0.4426 | 0.4393 | Pattern logits hurt noncentral extent |

## Combined System Comparison

Frozen references:

- `hgb_fallback`: BA `0.3280`, macro F1 `0.3327`
- best SEV-S1: `severity5__posthoc__geom_plus_patternlogits_hgb_v1__holdout_v1`
  - BA `0.3993`
  - macro F1 `0.4020`
- strict reference:
  - `severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_tempered_v1__holdout_v1__seed42`
  - BA `0.6109`
  - macro F1 `0.5542`

| Combined System | BA | Macro F1 | No-Ulcer Precision | Central Recall | Adjacent Error | dBA vs fallback | dBA vs SEV-S1 best | dBA vs strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `severity5__posthoc__factorized_geomrules_v1__holdout_v1` | 0.2757 | 0.2005 | 0.1600 | 0.3077 | 0.4588 | -0.0523 | -0.1236 | -0.3352 |
| `severity5__posthoc__factorized_geom_hgb_v1__holdout_v1` | 0.3863 | 0.4039 | 0.4000 | 0.3077 | 0.6667 | +0.0583 | -0.0130 | -0.2246 |
| `severity5__posthoc__factorized_geom_plus_patternlogits_hgb_v1__holdout_v1` | 0.4213 | 0.4347 | 1.0000 | 0.6923 | 0.5741 | +0.0933 | +0.0220 | -0.1896 |

## Best Combined Run

### `severity5__posthoc__factorized_geom_plus_patternlogits_hgb_v1__holdout_v1`

- balanced accuracy `0.4213`
- macro F1 `0.4347`
- no-ulcer precision `1.0000`
- central-ulcer recall `0.6923`
- adjacent-class error rate `0.5741`

Per-class recall / F1:

- `no_ulcer`: `0.1429 / 0.2500`
- `ulcer_leq_25pct`: `0.1333 / 0.1290`
- `ulcer_leq_50pct`: `0.5333 / 0.5246`
- `ulcer_geq_75pct`: `0.6047 / 0.5778`
- `central_ulcer`: `0.6923 / 0.6923`

Confusion matrix:

- labels: `no_ulcer, ulcer_leq_25pct, ulcer_leq_50pct, ulcer_geq_75pct, central_ulcer`
- matrix: `[[1, 3, 2, 1, 0], [0, 2, 4, 8, 1], [0, 2, 16, 11, 1], [0, 7, 8, 26, 2], [0, 2, 1, 1, 9]]`

Mild-bin / no-ulcer diagnostics:

- `ulcer_leq_25pct` prediction breakdown: `{'central_ulcer': 1, 'ulcer_geq_75pct': 8, 'ulcer_leq_25pct': 2, 'ulcer_leq_50pct': 4}`
- `ulcer_leq_50pct` prediction breakdown: `{'central_ulcer': 1, 'ulcer_geq_75pct': 11, 'ulcer_leq_25pct': 2, 'ulcer_leq_50pct': 16}`
- no-ulcer false negatives: `6`
- no-ulcer false positives: `0`

Interpretation:

- The factorized plus-pattern system is the best post-hoc severity result in the repo so far.
- The win comes from much cleaner S1 routing and a much cleaner `no_ulcer` decision boundary at precision level.
- It still does not solve the mild extent problem. `ulcer_leq_25pct` remains weak and many mild or medium cases are still escalated to `ulcer_geq_75pct`.

## Verdict

- Branch verdict: `useful but incomplete`
- Why:
  - It beats the best SEV-S1 run on both balanced accuracy and macro F1.
  - It materially improves no-ulcer precision from `0.25` to `1.00`.
  - It reduces adjacent-class error from `0.6429` to `0.5741`.
  - It narrows the gap to the strict severity reference.
  - It does not fix `ulcer_leq_25pct`, and S2 remains the limiting stage.
