# Dual Crop Deployment Fusion

| Strategy | Weights (Off, w35, Dual) | Val BA | Val F1 | Test BA | Test F1 | Test Acc | PL Recall | PFM Recall | Flaky Recall |
|---|---|---|---|---|---|---|---|---|---|
| Official Only | [np.float64(1.0), np.float64(0.0), np.float64(0.0)] | 0.7253 | 0.6968 | 0.8482 | 0.7990 | 0.8426 | 0.9630 | 0.6585 | 0.9231 |
| w0035 Only | [np.float64(0.0), np.float64(1.0), np.float64(0.0)] | 0.7030 | 0.7039 | 0.8671 | 0.8546 | 0.8796 | 0.9259 | 0.8293 | 0.8462 |
| Dual Crop Only | [np.float64(0.0), np.float64(0.0), np.float64(1.0)] | 0.6994 | 0.6991 | 0.7061 | 0.7118 | 0.7870 | 0.8519 | 0.8049 | 0.4615 |
| Equal Average | [np.float64(0.33), np.float64(0.33), np.float64(0.33)] | 0.7182 | 0.7169 | 0.8252 | 0.8092 | 0.8519 | 0.9259 | 0.7805 | 0.7692 |
| w0035=2, Dual=1 | [np.float64(0.0), np.float64(0.67), np.float64(0.33)] | 0.6967 | 0.6973 | 0.8077 | 0.8030 | 0.8519 | 0.9259 | 0.8049 | 0.6923 |
| Official=1, w0035=2, Dual=1 | [np.float64(0.25), np.float64(0.5), np.float64(0.25)] | 0.7030 | 0.7039 | 0.8333 | 0.8224 | 0.8611 | 0.9259 | 0.8049 | 0.7692 |
| Val-Selected Weights | [np.float64(0.75), np.float64(0.12), np.float64(0.12)] | 0.7447 | 0.7209 | 0.8645 | 0.8242 | 0.8611 | 0.9630 | 0.7073 | 0.9231 |