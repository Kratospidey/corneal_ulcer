# TG + Severity Rescue Results

## Executed work

### Run 1: TG-A3

| Field | Value |
| --- | --- |
| Experiment name | `tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42` |
| Config | `configs/train_tg_structured_t123_balsoftmax_t3.yaml` |
| Seed | `42` |
| Warm start | official pattern checkpoint |
| Training status | completed |
| Best validation balanced accuracy | `0.5126681449` |
| Best validation macro F1 | `0.5389632107` |
| Final test balanced accuracy | `0.3714285714` |
| Final test macro F1 | `0.3527685493` |
| Punctate-family balanced accuracy | `0.1388888889` |
| Punctate-family macro F1 | `0.1960784314` |
| Type3 recall | `0.0` |
| Type3 F1 | `0.0` |
| Type4 recall guardrail | `0.9404761905` |
| Verdict | `non-promotable` |

### TG-A3 classwise recall / F1

| Class | Recall | F1 |
| --- | ---: | ---: |
| `no_ulcer` | `0.5000` | `0.4286` |
| `micro_punctate` | `0.4167` | `0.4167` |
| `macro_punctate` | `0.0000` | `0.0000` |
| `coalescent_macro_punctate` | `0.0000` | `0.0000` |
| `patch_gt_1mm` | `0.9405` | `0.9186` |

### TG-A3 confusion summary

- `micro_punctate -> patch_gt_1mm`: `4`
- `macro_punctate -> patch_gt_1mm`: `4`
- `coalescent_macro_punctate -> patch_gt_1mm`: `1`
- `patch_gt_1mm` stayed strong, but punctate subclasses still collapsed into `patch_gt_1mm` or `no_ulcer`.

### TG-A3 promotability decision

- `type3` remained unlearned.
- `macro_punctate` also stayed at zero recall.
- Punctate-family metrics are too weak to justify extra seeds.
- The result is materially worse than the documented prior TG rescue headline (`TG-A1_serious_distill` BA `0.5102`, macro F1 `0.4920`).
- This did preserve the `type4` guardrail, but that alone is not enough.

### Main failure mode

- The hierarchy did not rescue the punctate tail.
- The model still defaults toward `patch_gt_1mm`.
- Validation looked better than test, so the old instability / mismatch story remains live.

## Required severity preflight

### Mask polarity sanity check

| Field | Value |
| --- | --- |
| Command | `PYTHONPATH=src micromamba run -n corneal-train python src/mask_polarity_debug.py --manifest-path /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/manifests/manifest.csv --split-file /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/split_files/pattern_3class_holdout.csv --output-dir outputs/debug/2026-04-22_mask_polarity_probe` |
| Status | completed |
| Samples inspected | `6` |
| Current matched corrected output for all samples | `True` |
| Interpretation | current cornea-mask polarity path is consistent with cornea-preserving masking |

### Ulcer supervision audit

| Field | Value |
| --- | --- |
| Command | `PYTHONPATH=src micromamba run -n corneal-train python src/run_ulcer_supervision_audit.py --manifest /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/manifests/manifest.csv --split-file /home/kratospidey/Repos/corneal_ulcer_classification/data/interim/split_files/pattern_3class_holdout.csv --output-root outputs/debug/2026-04-22_ulcer_supervision_audit` |
| Status | completed |
| Total audited images | `712` |
| Historical ulcer masks present | `354` |
| Candidate positive cases missing lesion masks | `322` |
| Candidate negatives needing explicit review | `36` |
| Needs review | `358` |

### Preflight conclusion

- Historical ulcer masks are incomplete auxiliary supervision.
- Missing masks are not negatives.
- Any canonical SEV-S1 pipeline must remain independent of mask absence semantics.

## What was not run

### TG-A3 extra seeds

- `seed73` not run.
- `seed17` not run.
- Reason: `seed42` failed the gate cleanly.

### TG-A4

- Not run.
- Reason: TG-A3 did not earn it. Running T3-only multiscale after `type3` stayed at zero would be guesswork.

### SEV-S1

- Not run.
- Reason: the clean freeze continuation branch does not contain a committed post-hoc severity feature pipeline.
- The available research-tree severity code is learned shared-backbone severity, which is explicitly not the recommended path.
- Building a proper post-hoc geometry + embedding severity stack from scratch here would be a substantial refactor and would violate the narrow continuation constraint.

## Artifact paths

- TG-A3 test metrics:
  `outputs/metrics/tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42/test_metrics.json`
- TG-A3 TG-focus metrics:
  `outputs/metrics/tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42/test_tg_focus_metrics.json`
- TG-A3 confusion matrix:
  `outputs/confusion_matrices/tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42/test_confusion_matrix.csv`
- Mask polarity probe:
  `outputs/debug/2026-04-22_mask_polarity_probe/summary.md`
- Ulcer supervision audit:
  `outputs/debug/2026-04-22_ulcer_supervision_audit/summary.md`

## Blunt conclusions

- What worked:
  - The official warm-start path is usable on the clean continuation branch.
  - The structured TG loss trained and evaluated cleanly.
  - `type4` remained strong.
  - The mask-polarity and ulcer-supervision guardrails were verified.

- What failed:
  - TG-A3 did not rescue `type3`.
  - TG-A3 also failed on `macro_punctate`.
  - Punctate-family metrics are too weak to justify expansion.
  - The documented post-unified TG evidence files referenced by the freeze docs are missing from the accessible workspace.

- What should be run next:
  - Do not run TG-A4 from this branch state.
  - Do not spend more seeds on this exact TG-A3 configuration.
  - If TG is revisited later, it needs a materially different but still narrow idea grounded in the missing rescue evidence, not just more repetitions of this setup.
  - SEV-S1 should only be attempted after a clean, explicit post-hoc feature pipeline is defined and committed.

- What should be stopped:
  - Stop TG-A3 extra seeds for this config.
  - Stop TG-A4 for this pass.
  - Stop any attempt to reinterpret missing historical ulcer masks as negatives.
  - Stop any drift back into shared-backbone learned severity as the default path.
