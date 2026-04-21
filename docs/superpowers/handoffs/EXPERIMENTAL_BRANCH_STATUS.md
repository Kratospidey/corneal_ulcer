# Experimental Branch Status

## TG rescue branch

- Status: promising but not promoted
- Best unrestricted result:
  - `TG-A1_serious_distill`
  - BA `0.5102`
  - macro F1 `0.4920`
- Best TG-safe result:
  - `TG-A2_multiscale_distill_seed42` guardrail checkpoint
  - BA `0.4970`
  - macro F1 `0.4500`
- Why not promoted:
  - seed sensitivity remains high
  - validation/test mismatch remains real
  - `type3` recall is still effectively zero
- Worth continuing: yes, but only as a narrow TG-focused line

## Severity salvage branch

- Status: secondary and unresolved
- Best salvage result:
  - `hgb_fallback`
  - BA `0.3280`
  - macro F1 `0.3327`
- Why not promoted:
  - no salvage model beat the prior best strict severity reference
  - learned geometry heads failed
  - severity remains weaker and less trustworthy than pattern
- Worth continuing: only as a post-hoc tabular / geometry line

## Unified 3-task branch

- Status: research artifact only
- Main artifact:
  - `pattern3_tg5_severity5__convnextv2_tiny__cornea_crop_scale_v1__unified_structured_tg_hybrid_severity_v1__holdout_v1__seed42`
- Why not promoted:
  - pattern regressed badly
  - learned severity branch did not work
  - no unified configuration justified replacing the official pattern path
- Worth continuing: not as the next broad pass

## Paper-style shadow benchmark branch

- Status: diagnostic only
- Best shadow winner:
  - ViT under the shadow protocol
  - pattern BA `0.6857`
  - TG BA `0.3407`
  - severity BA `0.3817`
- Why not promoted:
  - it is not comparable to the official leakage-safe benchmark
  - it did not demonstrate that the strict protocol is the main bottleneck
- Worth continuing: only for targeted methodology diagnosis, not as an official leaderboard
