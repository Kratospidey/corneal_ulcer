# corneal_ulcer_classification

This repo is frozen as a pattern-first classification project.

Final active benchmark line:

- official single model:
  - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - test balanced accuracy `0.8482`
  - test macro F1 `0.7990`

Best deployed inference rule:

- `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
  - test balanced accuracy `0.8563`
  - test macro F1 `0.8115`

Archived lines:

- TG / type
- severity / grade

They remain in the repo only as historical research artifacts. They are not active continuation targets.

Start with:

- [codex.md](codex.md)
- [FINAL_MODELING_STATE_2026-04-22.md](docs/superpowers/handoffs/FINAL_MODELING_STATE_2026-04-22.md)
- [FINAL_PATTERN_ONLY_DECISION.md](docs/superpowers/handoffs/FINAL_PATTERN_ONLY_DECISION.md)

Notes:

- the dataset is not included in this repo
- the official pattern checkpoint and the deployed late-fusion rule are distinct
- future presentation should center pattern only
