# Codex Working Truth

## Final Project State

- Date frozen: `2026-04-22`
- Official active task family: `pattern_3class`
- Archived task families:
  - `task_tg_5class`
  - `severity_5class`

## Pattern

- Official single model:
  - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - test balanced accuracy `0.8482`
  - test macro F1 `0.7990`
- Best deployed inference rule:
  - `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
  - test balanced accuracy `0.8563`
  - test macro F1 `0.8115`
- These remain distinct:
  - the official checkpoint is the benchmark model
  - the late-fusion rule is the best deployed inference stack

## TG / Type

- Archived / abandoned on this foundation.
- Final failed continuation:
  - `tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42`
  - test balanced accuracy `0.3714`
  - test macro F1 `0.3528`
- Final verdict:
  - data scarcity
  - hierarchy starvation
  - smaller label-boundary ambiguity
  - not a loss-tweak problem

## Severity / Grade

- Archived / abandoned on this foundation.
- Best post-hoc fallback:
  - `severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1`
  - test balanced accuracy `0.4233`
  - test macro F1 `0.4366`
- SEV-S3 no-ulcer precision `1.0000` was audited and found clean but extremely conservative:
  - test support `7`
  - predicted no-ulcer count `1`
  - no leakage, route leakage, or combiner bug found
- SEV-S4 stop result:
  - no variant beat SEV-S3 cleanly
  - any gain came from breaking already-useful gates or losing too much macro performance

## What Not To Do Next

- Do not reopen small TG loss tweaks.
- Do not reopen small severity tabular / routing / calibration tweaks.
- Do not destabilize the official pattern line.
- Do not present TG or severity as healthy benchmark lines.

## If Research Is Reopened Later

- It needs new signal, not more micro-tweaks.
- Realistic broader bets:
  - better lesion proxy / segmentation signal
  - label reform / bin reform
  - more or better supervision

## 2026-04-23 Post-Freeze Follow-Up

- No clean run exceeded `0.90` balanced accuracy.
- Best completed reproducible post-freeze artifact:
  - `pattern3__cornea_crop_scale_v1__convnextv2_tiny_plus_vit_small__stats__nearest_centroid__holdout_v1__seed42`
  - val balanced accuracy `0.7717`
  - test balanced accuracy `0.8743`
  - test macro F1 `0.8586`
- Important qualification:
  - this is an exploratory frozen-feature artifact, not an official benchmark replacement
  - the measured uplift depends on tabular mask stats, including ulcer-mask-derived coverage features
  - do not treat it as a clean image-only ConvNeXtV2 continuation line

## Segmentation-Assisted Pattern Extension

- A new pattern-only experimental track is allowed:
  - `image -> predicted cornea/ulcer masks -> predicted-mask-derived stats -> pattern_3class classifier`
- Promotion bar:
  - predicted masks only in the downstream classifier path
  - same frozen segmenter inference path for train, val, and test features
- Non-promotable:
  - any downstream experiment that uses human ulcer-mask-derived stats directly
- Current implementation surface:
  - `src/main_train_segmentation.py`
  - `src/main_eval_segmentation.py`
  - `src/main_infer_segmentation.py`
  - `src/main_extract_predicted_mask_stats.py`
  - `src/experimental/pattern/train_predmask_classifier.py`
- Current handoff docs:
  - `docs/superpowers/handoffs/SEGMENTATION_PATTERN_DATA_AUDIT.md`
  - `docs/superpowers/handoffs/SEGMENTATION_ASSISTED_PATTERN_PLAN.md`
  - `docs/superpowers/handoffs/SEGMENTATION_ASSISTED_PATTERN_RESULTS.md`

## 2026-04-23 Swin Ensemble Follow-Up

- Clean image-only partner tested:
  - `pattern3__swin_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
- Clean ensemble lines tested:
  - `pattern3__convnextv2_tiny_plus_swin_tiny__avgprob_eq__holdout_v1__seed42`
  - `pattern3__convnextv2_tiny_plus_swin_tiny__avgprob_valtuned__holdout_v1__seed42`
- Result:
  - valid negative result
  - no Swin-based ensemble beat the frozen official pattern benchmark or the deployed late-fusion rule
- Current handoff docs:
  - `docs/superpowers/handoffs/SWIN_ENSEMBLE_PATTERN_PLAN.md`
  - `docs/superpowers/handoffs/SWIN_ENSEMBLE_PATTERN_RESULTS.md`

## 2026-04-23 MaxViT Ensemble Follow-Up

- Clean image-only partner tested:
  - `pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
- Clean ensemble lines tested:
  - `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_eq__holdout_v1__seed42`
  - `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_valtuned__holdout_v1__seed42`
- Best new clean MaxViT ensemble:
  - `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_valtuned__holdout_v1__seed42`
  - val balanced accuracy `0.7254`
  - test balanced accuracy `0.8265`
  - test macro F1 `0.7943`
- Result:
  - valid negative result
  - closer than the Swin-based ensemble follow-up, but still below the frozen official single-model benchmark and the deployed late-fusion rule
- Current handoff docs:
  - `docs/superpowers/handoffs/MAXVIT_ENSEMBLE_PATTERN_PLAN.md`
  - `docs/superpowers/handoffs/MAXVIT_ENSEMBLE_PATTERN_RESULTS.md`
