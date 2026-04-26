# Post-DualCrop Recovery Results

## Anchor Table

| Run | BA | Macro F1 | Accuracy | PL Recall | PFM Recall | Flaky Recall |
|---|---|---|---|---|---|---|
| Official anchor | 0.8482 | 0.7990 | 0.8426 | 0.9630 | 0.6585 | 0.9231 |
| w0035 current best | 0.8671 | 0.8546 | 0.8796 | 0.9259 | 0.8293 | 0.8462 |
| best crop-consistency deployment rule | 0.8671 | 0.8546 | 0.8796 | 0.9259 | 0.8293 | 0.8462 |
| best new seed checkpoint (seed 7) | 0.8427 | 0.8251 | 0.8519 | 0.9259 | 0.7561 | 0.8462 |
| best checkpoint soup (all 5 seeds) | 0.8509 | 0.8273 | 0.8611 | 0.9259 | 0.7805 | 0.8462 |
| best logit soup (w0035_seed42 + w0035_seed21) | 0.8590 | 0.8351 | 0.8704 | 0.9259 | 0.8049 | 0.8462 |

## Dual-Crop Negative Finding Summary
Dual-crop context models (D0-D4) failed to improve upon the w0035 baseline. The highest Test BA reached was 0.8171. The expanded classification head suffered from a collapse in flaky recall, likely due to a lack of sufficient training data to learn the minority class boundaries with the increased parameter count in the fusion stage. This architecture is not recommended for promotion.

## Crop-Consistency Results
Evaluating the frozen w0035 checkpoint on multiple deterministic crop variants (base, slightly wide, slightly tight, shifted) did not meaningfully improve performance. Most combinations degraded performance or perfectly matched the base w0035 performance (BA 0.8671).

## Seed Confirmation Results
Four additional seeds (1, 7, 21, 123) were run using the exact w0035 recipe to verify its stability. None of the new seeds reached the 0.8671 Test BA of the original seed 42. Seed 7 was the closest at 0.8427 Test BA. The original w0035 seed 42 appears to represent a particularly strong (and perhaps slightly lucky) convergence trajectory.

## Checkpoint Soup Results
Creating a uniform average of the model weights across the 5 seeds ("Uniform_All5") or the top 3 validation seeds ("Uniform_Top3_Val") resulted in a Test BA of 0.8509. While this is strong and stable, it does not beat the single best w0035 checkpoint.

## Logit Soup Results
Averaging the output logits of different combinations of models also failed to surpass the w0035 baseline. The best combination (w0035_seed42 + w0035_seed21) achieved a Test BA of 0.8590. Even a validation-selected weighted blend of the Official and w0035 models peaked at 0.8563.

## Conclusion and Recommendation
**w0035 remains the best single checkpoint and the recommended deployment model.**

No single-checkpoint, logit soup, checkpoint soup, or crop-consistency rule managed to exceed the 0.8671 Test BA and 0.8546 Macro F1 established by w0035. The 90%+ BA target was not reached.

**Recommended next step:** External-data self-supervised pretraining or carefully selected additional human-labeled data is the next major direction to pursue. The current architecture and limited dataset size appear to have reached their practical limit for simple boundary refinement.
