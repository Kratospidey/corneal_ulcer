# Logit Boundary Tuning

Validation selected the decision rule by balanced accuracy first, then macro F1.

- Best validation temperature by NLL: 2.1000
- Best validation class bias: {'point_like': 0.0, 'point_flaky_mixed': -0.25, 'flaky': 0.6}

| Variant | Val BA | Val Macro F1 | Val ECE | Test BA | Test Macro F1 | Test ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.7253 | 0.6968 | 0.1161 | 0.8482 | 0.7990 | 0.0790 |
| temperature_only | 0.7253 | 0.6968 | 0.0896 | 0.8482 | 0.7990 | 0.0656 |
| bias_only | 0.7734 | 0.7138 | 0.1633 | 0.8157 | 0.7505 | 0.1256 |
| bias_plus_temperature | 0.7734 | 0.7138 | 0.1174 | 0.8157 | 0.7505 | 0.0831 |