# w0035 Seed Confirmation Results

| seed | val_BA | val_macro_F1 | test_BA | test_macro_F1 | accuracy | PL_recall | PFM_recall | flaky_recall | checkpoint_path | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.7030 | 0.7039 | 0.8671 | 0.8546 | 0.8796 | 0.9259 | 0.8293 | 0.8462 | models/exported/..._seed42/best.pt | w0035 baseline |
| 1 | 0.7012 | 0.6972 | 0.8333 | 0.8224 | 0.8611 | 0.9259 | 0.8049 | 0.7692 | models/exported/..._seed1/best.pt | |
| 7 | 0.6778 | 0.6715 | 0.8427 | 0.8251 | 0.8519 | 0.9259 | 0.7561 | 0.8462 | models/exported/..._seed7/best.pt | |
| 21 | 0.7348 | 0.6973 | 0.8203 | 0.7794 | 0.8241 | 0.9074 | 0.7073 | 0.8462 | models/exported/..._seed21/best.pt | Highest val BA |
| 123 | 0.7056 | 0.6946 | 0.8245 | 0.8010 | 0.8333 | 0.9444 | 0.6829 | 0.8462 | models/exported/..._seed123/best.pt | |