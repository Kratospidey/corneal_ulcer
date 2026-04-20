# Model Readiness Summary

- Dataset rows available for baseline classification: 712
- Resolved device for feature extraction/training: cuda
- Pattern imbalance ratio: 3.934066
- Severity imbalance ratio: 7.583333
- TG imbalance ratio: 54.8
- Ulcer-mask subset ratio: 0.497191

## Recommendation

- Start Stage 3 with raw RGB + convnextv2_tiny on the confirmed 3-class pattern task.
- Run masked_highlight_proxy as the matched paper-inspired baseline rather than assuming preprocessing is superior.
- Keep imbalance-aware metrics and duplicate-aware split validation mandatory.