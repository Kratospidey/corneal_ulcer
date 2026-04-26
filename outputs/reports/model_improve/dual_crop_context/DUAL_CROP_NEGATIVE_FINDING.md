# Dual-Crop Context Negative Finding

- Dual-crop was tested after w0035 to provide wide peripheral context for the model.
- Five configs were run: D0, D1, D2, D3, D4.
- Best dual-crop test BA was 0.8171.
- w0035 remains the best single checkpoint at 0.8671 BA.
- Deployment fusion of official, w0035, and dual-crop models did not beat w0035.
- Flaky recall collapsed in dual-crop models.
- Conclusion: Do not promote or continue this dual-crop architecture for now.
- Recommendation: Move to crop-consistency inference and w0035-style checkpoint/logit soups.
