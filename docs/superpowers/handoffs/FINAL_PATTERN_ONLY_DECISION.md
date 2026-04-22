# Final Pattern-Only Decision

- The repo is now formally pattern-first.
- `pattern_3class` is the only active healthy benchmark line.
- `task_tg_5class` and `severity_5class` are archived result lines, not current continuation targets.
- Future repo presentation, README wording, `codex.md`, and handoff docs should center pattern only.
- The official pattern model and the deployed inference rule remain distinct:
  - official benchmark checkpoint:
    - `pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
  - best deployed inference rule:
    - `pattern3__convnextv2_tiny__crop_scale_raw_multiscale__latefusion_v1__holdout_v1__seed42combo`
- Do not casually present TG or severity as viable benchmark lines.
- Do not blur archived TG / severity artifacts into the main repo story.

