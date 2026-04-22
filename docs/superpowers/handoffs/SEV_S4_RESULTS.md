# SEV-S4 Final Narrow Severity Continuation

## Scope

- Pattern stayed frozen.
- TG stayed paused.
- Severity remained fully post-hoc.
- Phase A audited SEV-S3 no-ulcer precision before any continuation.
- Phase B only tested ultra-narrow calibration / probability-coupling / conflict-handling changes.
- No new backbone training, no feature branch expansion, no TG work, no broad redesign.

## Files Added Or Modified

- `src/experimental/severity/audit_no_ulcer_precision.py`
- `src/experimental/severity/eval_s4_calibrated_severity.py`
- `docs/superpowers/handoffs/SEV_S3_NO_ULCER_AUDIT.md`
- `docs/superpowers/handoffs/SEV_S4_RESULTS.md`
- `docs/superpowers/handoffs/WORKING_SEV_S4_NOTE.md`

## Commands Run

```bash
micromamba run -n corneal-train python -V
micromamba run -n corneal-train python -c "import torch, sklearn; print(torch.__version__); print(torch.cuda.is_available()); print(sklearn.__version__)"
python -m py_compile src/experimental/severity/audit_no_ulcer_precision.py src/experimental/severity/eval_s4_calibrated_severity.py
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/audit_no_ulcer_precision.py \
  --geom-table outputs/debug/severity_posthoc/geom_table_v3.csv \
  --manifest /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/manifests/manifest.csv \
  --split-file /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/split_files/pattern_3class_holdout.csv \
  --final-experiment severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1 \
  --s0-experiment severity5__posthoc__factorized_s0_plus_patternlogits_hgb_v1__holdout_v1 \
  --output-root outputs \
  --output-dir outputs/debug/severity_posthoc/sev_s3_no_ulcer_audit \
  --report-path docs/superpowers/handoffs/SEV_S3_NO_ULCER_AUDIT.md
```

```bash
PYTHONPATH=src micromamba run -n corneal-train python src/experimental/severity/eval_s4_calibrated_severity.py \
  --geom-table outputs/debug/severity_posthoc/geom_table_v3.csv \
  --s0-experiment severity5__posthoc__factorized_s0_plus_patternlogits_hgb_v1__holdout_v1 \
  --s1-experiment severity5__posthoc__factorized_s1_plus_patternlogits_hgb_v1__holdout_v1 \
  --s2-leq25-experiment severity5__posthoc__factorized_s2_leq25_hgb_v1__holdout_v1 \
  --s2-geq75-experiment severity5__posthoc__factorized_s2_geq75_hgb_v1__holdout_v1 \
  --s0cal-experiment-name severity5__posthoc__factorized_geom_plus_patternlogits_s0cal_hgb_v1__holdout_v1 \
  --s2cal-experiment-name severity5__posthoc__factorized_geom_plus_patternlogits_s2cal_hgb_v1__holdout_v1 \
  --probcombine-experiment-name severity5__posthoc__factorized_geom_plus_patternlogits_probcombine_hgb_v1__holdout_v1 \
  --output-root outputs \
  --report-path docs/superpowers/handoffs/SEV_S4_RESULTS.md
```

## Phase A Audit Findings

Audit artifact:

- `docs/superpowers/handoffs/SEV_S3_NO_ULCER_AUDIT.md`

Blunt answer:

- The `1.0000` no-ulcer precision is trustworthy, but it is not robust.
- It comes from extremely conservative behavior on tiny predicted count, not from a hidden win.

Exact audit findings on the frozen test split:

- true `no_ulcer` support: `7`
- predicted `no_ulcer` count: `1`
- `no_ulcer` precision: `1.0000`
- `no_ulcer` recall: `0.1429`
- `no_ulcer` F1: `0.2500`
- correct predicted `no_ulcer` id: `['324']`
- missed true `no_ulcer` ids: `['175', '280', '308', '36', '89', '94']`
- false `no_ulcer` prediction ids: `[]`

S0-specific audit:

- S0 confusion matrix: `[[1, 6], [0, 101]]`
- S0 predicted `no_ulcer` count on test: `1`
- final predicted `no_ulcer` count exactly matches S0 `no_ulcer` count: `True`

Leakage / bug checks:

- duplicate image ids after merge: `0`
- raw image paths crossing splits: `0`
- cornea mask paths crossing splits: `0`
- suspicious numeric feature columns: `[]`
- forbidden feature columns in S0 training features: `[]`
- pattern features limited to logits / probs / confidence / interactions: `True`
- evaluator route leakage via `factorized_route`: `False`

Audit verdict:

- `clean conservative behavior`

## SEV-S4 Variants

Baseline carry-forward:

- `severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1`

New ultra-narrow variants:

- `severity5__posthoc__factorized_geom_plus_patternlogits_s0cal_hgb_v1__holdout_v1`
- `severity5__posthoc__factorized_geom_plus_patternlogits_s2cal_hgb_v1__holdout_v1`
- `severity5__posthoc__factorized_geom_plus_patternlogits_probcombine_hgb_v1__holdout_v1`

Calibration fits used validation only:

- `s0_no_ulcer`: class balance `0.0377`, fitted `True`
- `s2_leq25`: class balance `0.1379`, fitted `True`
- `s2_geq75`: class balance `0.4138`, fitted `True`

## SEV-S4 Comparison Table

| Experiment | BA | Macro F1 | No-Ulcer Precision | No-Ulcer Recall | Central Recall | leq25 Recall | leq25 F1 | Adjacent Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1` | 0.4233 | 0.4366 | 1.0000 | 0.1429 | 0.6923 | 0.2000 | 0.1765 | 0.6727 |
| `severity5__posthoc__factorized_geom_plus_patternlogits_s0cal_hgb_v1__holdout_v1` | 0.4948 | 0.4268 | 0.2727 | 0.8571 | 0.6154 | 0.0667 | 0.0769 | 0.5690 |
| `severity5__posthoc__factorized_geom_plus_patternlogits_s2cal_hgb_v1__holdout_v1` | 0.3892 | 0.3823 | 1.0000 | 0.1429 | 0.6923 | 0.2667 | 0.2051 | 0.5593 |
| `severity5__posthoc__factorized_geom_plus_patternlogits_probcombine_hgb_v1__holdout_v1` | 0.4561 | 0.3580 | 0.2143 | 0.8571 | 0.6154 | 0.0667 | 0.0690 | 0.5000 |

## Exact Comparison Against References

Frozen references:

- `hgb_fallback`: BA `0.3280`, macro F1 `0.3327`
- best SEV-S1: BA `0.3993`, macro F1 `0.4020`
- best SEV-S2: BA `0.4213`, macro F1 `0.4347`
- best SEV-S3: BA `0.4233`, macro F1 `0.4366`
- strict severity reference: BA `0.6109`, macro F1 `0.5542`

Key deltas:

- `s0cal` vs best SEV-S3:
  - `+0.0715` BA
  - `-0.0097` macro F1
  - but no-ulcer precision collapses from `1.0000` to `0.2727`
  - central-ulcer recall drops from `0.6923` to `0.6154`
  - `ulcer_leq_25pct` recall drops from `0.2000` to `0.0667`
- `s2cal` vs best SEV-S3:
  - `-0.0341` BA
  - `-0.0543` macro F1
  - `ulcer_leq_25pct` recall improves from `0.2000` to `0.2667`
  - `ulcer_leq_25pct` F1 improves from `0.1765` to `0.2051`
  - adjacent error improves from `0.6727` to `0.5593`
  - but overall system quality drops too much
- `probcombine` vs best SEV-S3:
  - `+0.0327` BA
  - `-0.0785` macro F1
  - no-ulcer precision collapses to `0.2143`
  - central-ulcer recall drops to `0.6154`

## Interpretation

- Phase A cleared SEV-S3: the perfect no-ulcer precision is genuine, but only because the system predicts `no_ulcer` once.
- Phase B did not produce a clean continuation.
- `s0cal` improves balanced accuracy by aggressively opening the no-ulcer gate, but that gain is not promotable because it breaks the gate behavior that made SEV-S3 acceptable.
- `s2cal` is the only variant that helps the mildest class without harming no-ulcer precision, but it loses too much overall BA and macro F1.
- `probcombine` improves BA and adjacent error, but it collapses both no-ulcer precision and macro F1.

## Blunt Verdict

- Branch verdict: `clean but no further gain`
- Continuation verdict: `stop severity continuation`

Why:

- The SEV-S3 no-ulcer precision is clean enough to trust.
- None of the SEV-S4 variants beats SEV-S3 cleanly.
- Every apparent gain comes from sacrificing gates that already worked or from losing too much macro performance elsewhere.
- The next idea would require a broader severity redesign than the allowed calibration / coupling scope.
