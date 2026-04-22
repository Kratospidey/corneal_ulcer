# Final TG Status

## Final Failed Continuation

- `tg5__convnextv2_tiny__officialwarm__structured_t123__balsoftmax_t3__holdout_v1__seed42`
- balanced accuracy `0.3714`
- macro F1 `0.3528`
- punctate-family balanced accuracy `0.1389`
- punctate-family macro F1 `0.1961`
- `type3` recall / F1 `0.0000 / 0.0000`
- `type4` recall `0.9405`

## Punctate Audit Verdict

- TG fails mainly because of:
  - data scarcity
  - hierarchy starvation
  - smaller label-boundary ambiguity
- The punctate-family audit did not support another small rescue pass.

## Why TG Is Abandoned

- `type3` support is too small to recover with incremental recipe tweaks.
- The structured TG line did not become competitively viable enough to justify more compute.
- The branch failure is not a simple loss-tweak problem.
- Running more seeds or tiny class-weight changes would not change the underlying limitation.

TG / type is abandoned as an active continuation line on this foundation.

