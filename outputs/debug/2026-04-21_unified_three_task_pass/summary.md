# Unified Three-Task Pass Summary

## What I verified
- Unified task audit written from the official holdout split.
- Required model ladder completed: A, B, C, D, E.
- Extra controls completed: one ImageNet-only init control on Model C, one CAM-proxy comparison on Model C.
- Results below come from saved run artifacts under this pass root, not terminal memory.

## Baseline used
- Official pattern single-model baseline test BA: 0.8482
- Best deployed pattern inference rule test BA: 0.8563
- Old severity single-model baseline test BA: 0.4585
- Prior hierarchical severity reference test BA: 0.6021

## Key findings
- Best unrestricted unified model by the configured TG validation rule: C_hybrid_image_proxy (val TG BA 0.7500, test TG BA 0.4789).
- Best validation-safe unified model: C_hybrid_image_proxy.
- Best held-out TG BA actually observed among all unified runs: C_cam_proxy_compare at 0.5651.
- Best held-out pattern BA from this pass: A_flat_multitask at 0.7964.
- Best held-out learned-severity BA from this pass: A_flat_multitask at 0.5662.
- Structured TG beat the flat TG baseline on validation and on held-out TG test for the simple structured model (B over A).
- TG multiscale fusion helped the fused TG readout relative to the plain TG readout inside D/E, but the overall D/E models did not beat C and never reached the guardrail.
- Mild TG label smoothing did not rescue the multiscale branch.
- Warm-start from the official pattern checkpoint clearly beat the matched ImageNet-only control.
- The learned geometry-aware severity head did not beat the flat severity classifier and usually collapsed to near-chance.
- The fixed image-proxy geometry rule baseline consistently beat the learned geometry heads, including the CAM comparison, but remained far below the prior severity reference.

## Selected model status
- Selected unrestricted model: pattern3_tg5_severity5__convnextv2_tiny__cornea_crop_scale_v1__unified_structured_tg_hybrid_severity_v1__holdout_v1__seed42
- Selected unrestricted held-out pattern BA delta vs official baseline: -0.1142
- Selected unrestricted held-out severity BA delta vs prior hierarchical reference: -0.4021
- Selected validation-safe model: pattern3_tg5_severity5__convnextv2_tiny__cornea_crop_scale_v1__unified_structured_tg_hybrid_severity_v1__holdout_v1__seed42
- Validation-safe held-out pattern BA delta vs official baseline: -0.1142

## Recommendation
- Do not change the official pattern single model.
- Do not change the best deployed pattern inference rule.
- Keep the warm-started Model C checkpoint as the main research artifact for this pass because it was the only serious architecture to produce a validation-safe checkpoint under the configured rule.
- Keep Model B in view because it had the strongest held-out TG BA among the non-control warm-started unified runs even though it did not pass the configured validation guardrail.
- Treat the geometry-aware severity branch as not yet successful. The rule baseline is more trustworthy than the learned geometry heads right now.
