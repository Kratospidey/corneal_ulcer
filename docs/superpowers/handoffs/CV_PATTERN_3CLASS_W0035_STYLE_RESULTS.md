# CV Pattern 3-Class w0035 Style Results

The fixed holdout result remains the frozen deployment benchmark. The new cross-validation benchmark estimates the robustness of the w0035-style training recipe across alternative stratified splits.

## Summary Results (10-Fold CV)
- **Mean Test Balanced Accuracy**: 0.7109 (Std: 0.0795)
- **Mean Test Macro F1**: 0.6555 (Std: 0.0749)
- **Mean Test Accuracy**: 0.6964 (Std: 0.0746)
- **PL Recall Mean**: 0.7513
- **PFM Recall Mean**: 0.5903
- **Flaky Recall Mean**: 0.7911

## Conclusion
The CV results (Mean Test BA: 0.7109) show that the w0035-style recipe's performance is significantly lower on average across random splits than on the specific fixed holdout split (Test BA: 0.8671). The high standard deviation (0.0795) and the collapse of the PFM recall in several folds suggest that the w0035 checkpoint's performance on the holdout set may be a particularly strong/lucky trajectory that is difficult to consistently reproduce on new random splits. The fixed holdout w0035 checkpoint remains the best deployment model, but the recipe itself lacks robustness.
