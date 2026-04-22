# SEV-S1 Post-Hoc Comparison

- Strict reference: `severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_tempered_v1__holdout_v1__seed42` with BA 0.6109 and macro F1 0.5542
- Fallback reference: `hgb_fallback` with BA 0.3280 and macro F1 0.3327

| Experiment | BA | Macro F1 | Central Recall | No-Ulcer Precision | Adjacent Error | dBA vs fallback | dMacro F1 vs fallback | dBA vs strict | dMacro F1 vs strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| severity5__posthoc__geom_plus_patternlogits_hgb_v1__holdout_v1 | 0.3993 | 0.4020 | 0.6923 | 0.2500 | 0.6429 | 0.0713 | 0.0692 | -0.2116 | -0.1523 |
| severity5__posthoc__geomrules_v1__holdout_v1 | 0.3688 | 0.2514 | 0.3846 | 0.1622 | 0.3924 | 0.0409 | -0.0813 | -0.2421 | -0.3028 |
| severity5__posthoc__geom_hgb_v1__holdout_v1 | 0.3278 | 0.3381 | 0.3077 | 0.2000 | 0.6719 | -0.0002 | 0.0054 | -0.2831 | -0.2161 |

## Per-Run Detail

### `severity5__posthoc__geom_plus_patternlogits_hgb_v1__holdout_v1`

- balanced accuracy `0.3993`
- macro F1 `0.4020`
- central-ulcer recall `0.6923`
- no-ulcer precision `0.2500`
- adjacent-class error rate `0.6429`
- per-class recall / F1
- `no_ulcer`: `0.1429 / 0.1818`
- `ulcer_leq_25pct`: `0.0667 / 0.0690`
- `ulcer_leq_50pct`: `0.4667 / 0.4667`
- `ulcer_geq_75pct`: `0.6279 / 0.6000`
- `central_ulcer`: `0.6923 / 0.6923`
- confusion matrix
  - labels: `no_ulcer, ulcer_leq_25pct, ulcer_leq_50pct, ulcer_geq_75pct, central_ulcer`
  - matrix: `[[1, 3, 1, 2, 0], [2, 1, 6, 5, 1], [1, 2, 14, 12, 1], [0, 6, 8, 27, 2], [0, 2, 1, 1, 9]]`

### `severity5__posthoc__geomrules_v1__holdout_v1`

- balanced accuracy `0.3688`
- macro F1 `0.2514`
- central-ulcer recall `0.3846`
- no-ulcer precision `0.1622`
- adjacent-class error rate `0.3924`
- per-class recall / F1
- `no_ulcer`: `0.8571 / 0.2727`
- `ulcer_leq_25pct`: `0.2667 / 0.3200`
- `ulcer_leq_50pct`: `0.0333 / 0.0588`
- `ulcer_geq_75pct`: `0.3023 / 0.3881`
- `central_ulcer`: `0.3846 / 0.2174`
- confusion matrix
  - labels: `no_ulcer, ulcer_leq_25pct, ulcer_leq_50pct, ulcer_geq_75pct, central_ulcer`
  - matrix: `[[6, 1, 0, 0, 0], [4, 4, 0, 2, 5], [10, 2, 1, 7, 10], [12, 3, 2, 13, 13], [5, 0, 1, 2, 5]]`

### `severity5__posthoc__geom_hgb_v1__holdout_v1`

- balanced accuracy `0.3278`
- macro F1 `0.3381`
- central-ulcer recall `0.3077`
- no-ulcer precision `0.2000`
- adjacent-class error rate `0.6719`
- per-class recall / F1
- `no_ulcer`: `0.1429 / 0.1667`
- `ulcer_leq_25pct`: `0.2000 / 0.1818`
- `ulcer_leq_50pct`: `0.5000 / 0.4839`
- `ulcer_geq_75pct`: `0.4884 / 0.4773`
- `central_ulcer`: `0.3077 / 0.3810`
- confusion matrix
  - labels: `no_ulcer, ulcer_leq_25pct, ulcer_leq_50pct, ulcer_geq_75pct, central_ulcer`
  - matrix: `[[1, 3, 2, 1, 0], [2, 3, 5, 5, 0], [0, 1, 15, 14, 0], [1, 7, 10, 21, 4], [1, 4, 0, 4, 4]]`
