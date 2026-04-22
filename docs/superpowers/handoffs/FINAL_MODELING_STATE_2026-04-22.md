# Final Modeling State 2026-04-22

## Final Decision

- Pattern is the only active and recommended task family.
- TG / type is abandoned as an active continuation line on this foundation.
- Severity / grade is abandoned as an active continuation line on this foundation.
- The repo should now be presented as pattern-first only.

## Final Winners

- Official pattern model:
  - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - balanced accuracy `0.8482`
  - macro F1 `0.7990`
- Best deployed inference rule:
  - `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
  - balanced accuracy `0.8563`
  - macro F1 `0.8115`

## Summary Table

| Line | Best experiment | Balanced accuracy | Macro F1 | Status | Verdict |
| --- | --- | ---: | ---: | --- | --- |
| official pattern | `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42` | 0.8482 | 0.7990 | frozen official | keep |
| deployed pattern rule | `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo` | 0.8563 | 0.8115 | deployed rule | keep distinct from official checkpoint |
| TG-A3 | `tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42` | 0.3714 | 0.3528 | abandoned | not competitively viable; not a loss-tweak problem |
| best SEV-S1 | `severity5__posthoc__geom_plus_patternlogits_hgb_v1__holdout_v1` | 0.3993 | 0.4020 | historical | beat fallback but far from strict severity reference |
| best SEV-S2 | `severity5__posthoc__factorized_geom_plus_patternlogits_hgb_v1__holdout_v1` | 0.4213 | 0.4347 | historical | useful improvement, still unresolved |
| best SEV-S3 | `severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1` | 0.4233 | 0.4366 | final severity fallback | clean but still unresolved |
| SEV-S4 stop result | `severity5__posthoc__factorized_geom_plus_patternlogits_s0cal_hgb_v1__holdout_v1` | 0.4948 | 0.4268 | rejected | higher BA came from breaking no-ulcer precision and gate quality |

## What Was Tried

Pattern:

- ConvNeXtV2 pattern training under the leakage-safe holdout
- crop / scale variants
- multiscale late-fusion deployment rule

TG:

- flat TG
- structured TG
- rescue continuation ending in TG-A3
- punctate-family diagnostic audit

Severity:

- post-hoc geometry rules
- post-hoc HGB
- factorized S0 / S1 / S2 routing
- ordinal S2 rescue
- no-ulcer audit
- ultra-narrow calibration / probability-coupling pass

## What Worked

- Pattern matured into the only stable line worth presenting.
- The official single pattern checkpoint is strong and reproducible.
- The separate deployed late-fusion rule improved pattern inference further.
- Severity post-hoc work produced a fallback improvement over `hgb_fallback`, but never a truly promotable grade line.

## What Failed

TG:

- punctate-family performance remained non-promotable
- `type3` remained effectively unsolved
- TG-A3 confirmed the branch was not recoverable with small tweaks

Severity:

- strict severity learning did not become a trustworthy continuation line
- post-hoc rescues improved some metrics but remained too weak overall
- SEV-S4 did not produce a clean improvement over SEV-S3

## What Was Stopped

- stop TG / type continuation on this foundation
- stop severity / grade continuation on this foundation
- stop small TG loss tweaks
- stop small severity tabular / routing / calibration tweaks
- keep future repo presentation centered on pattern only

