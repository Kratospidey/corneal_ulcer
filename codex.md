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

