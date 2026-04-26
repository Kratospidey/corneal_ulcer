# External SSL Fine-Tune Results

| run | pretrain source | init mode | ordinal weight | val BA | val macro F1 | test BA | test macro F1 | accuracy | PL recall | PFM recall | flaky recall | beats w0035 | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w0035 | ImageNet | official_checkpoint | 0.035 | 0.7030 | 0.7039 | 0.8671 | 0.8546 | 0.8796 | 0.9259 | 0.8293 | 0.8462 | baseline | Current best |
| SSL-FT1 | SLID+SLITNet (SimSiam 30ep) | external_backbone | 0.035 | 0.6159 | 0.4571 | 0.6274 | 0.4450 | 0.5556 | 0.8333 | 0.0488 | 1.0000 | No | Collapsed on PFM |
| SSL-FT2 | SLID+SLITNet (SimSiam 30ep) | external_backbone | 0.025 | 0.6159 | 0.4571 | 0.6274 | 0.4450 | 0.5556 | 0.8333 | 0.0488 | 1.0000 | No | Collapsed on PFM |