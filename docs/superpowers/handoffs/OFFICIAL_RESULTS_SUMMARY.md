# Official Results Summary

## Official model

- `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
- task: `pattern_3class`
- test BA: `0.8482`
- test macro F1: `0.7990`

## Best deployed inference rule

- `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
- task: `pattern_3class`
- test BA: `0.8563`
- test macro F1: `0.8115`
- this is an inference rule, not the official single checkpoint

## Current task status

- Pattern: mature, strongest, recommended for real use
- TG: experimental; structured TG looks better than flat TG, but `type3` is still unresolved and the branch is not stable enough to promote
- Severity: experimental / post-hoc; best salvage baseline is `hgb_fallback`, but no salvage model beat the prior best strict severity reference

## What a collaborator should use right now

- For official reporting: use the official pattern checkpoint
- For best current pattern inference: use the deployed late-fusion rule
- Do not treat TG or severity as official outputs yet
- Do not treat unified or shadow results as production guidance
