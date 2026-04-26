# Crop Consistency Results

| run | view_set | aggregation | val_BA | val_macro_F1 | test_BA | test_macro_F1 | accuracy | PL_recall | PFM_recall | flaky_recall | beats_w0035 | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w0035 | Set A (base, slightly_wide, slightly_tight) | logit_average | 0.7030 | 0.7039 | 0.8671 | 0.8616 | 0.8796 | 0.9259 | 0.8293 | 0.8462 | Yes |  |
| w0035 | Set A (base, slightly_wide, slightly_tight) | prob_average | 0.7030 | 0.7039 | 0.8671 | 0.8546 | 0.8796 | 0.9259 | 0.8293 | 0.8462 | Yes |  |
| w0035 | Set B (base, wide_context) | logit_average | 0.6792 | 0.6793 | 0.8415 | 0.8361 | 0.8704 | 0.9259 | 0.8293 | 0.7692 | No |  |
| w0035 | Set B (base, wide_context) | prob_average | 0.6792 | 0.6793 | 0.8415 | 0.8361 | 0.8704 | 0.9259 | 0.8293 | 0.7692 | No |  |
| w0035 | Set C (base, slightly_wide, shift_left_up, shift_right_down) | logit_average | 0.6792 | 0.6793 | 0.8671 | 0.8546 | 0.8796 | 0.9259 | 0.8293 | 0.8462 | Yes |  |
| w0035 | Set C (base, slightly_wide, shift_left_up, shift_right_down) | prob_average | 0.6792 | 0.6793 | 0.8671 | 0.8546 | 0.8796 | 0.9259 | 0.8293 | 0.8462 | Yes |  |