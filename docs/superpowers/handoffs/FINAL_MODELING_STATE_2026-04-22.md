# Final Modeling State 2026-04-22

## Final Decision

- Pattern is the only active and recommended task family.
- TG / type and severity / grade are historical stop lines, not active continuations.
- The repo should be presented as pattern-only.

## Final Winners

- Official pattern model:
  - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - balanced accuracy `0.8482`
  - macro F1 `0.7990`
- Best deployed inference rule:
  - `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
  - balanced accuracy `0.8563`
  - macro F1 `0.8115`

## Live Scope

| Line | Best experiment | Balanced accuracy | Macro F1 | Status |
| --- | --- | ---: | ---: | --- |
| official pattern | `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42` | 0.8482 | 0.7990 | frozen official |
| deployed pattern rule | `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo` | 0.8563 | 0.8115 | deployed rule |

## What Worked

- Pattern matured into the only stable line worth presenting.
- The official single pattern checkpoint is strong and reproducible.
- The separate deployed late-fusion rule improved pattern inference further.

## What Was Removed From The Live Repo Surface

- TG / type continuation code
- severity / grade continuation code
- segmentation-assisted and proxy-geometry branches
- external warm-start and exploratory larger-backbone branches
