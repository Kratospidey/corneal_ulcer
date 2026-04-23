# MaxViT Ensemble Pattern Results

## Scope

- Active task only: `pattern_3class`
- New single-model line:
  - `pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42`
- New ensemble lines:
  - `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_eq__holdout_v1__seed42`
  - `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_valtuned__holdout_v1__seed42`
- Explicitly excluded:
  - TG/type
  - severity/grade
  - segmentation
  - mask-derived features
  - extra ensemble variants beyond the planned equal-weight and one validation-tuned rule

## Live Pipeline Assumptions Used

- Canonical split:
  - `data/interim/split_files/pattern_3class_holdout.csv`
  - split counts:
    - train `498`
    - val `106`
    - test `108`
- Label order:
  - `point_like`
  - `point_flaky_mixed`
  - `flaky`
- Shared crop and augmentation recipe:
  - `cornea_crop_scale_v1`
  - `pattern_augplus_v2`
- Shared sampling rule:
  - `weighted_sampler_tempered`
  - `sampler_temperature: 0.65`
- Shared checkpoint-selection metric:
  - validation `balanced_accuracy`
- Shared prediction export contract:
  - `image_id`
  - `split`
  - `target_index`
  - `predicted_index`
  - `prob_point_like`
  - `prob_point_flaky_mixed`
  - `prob_flaky`
  - `<split>_prediction_metadata.json` with fixed class-order provenance

## Runs Actually Executed

### MaxViT-Tiny single model

- Train command:
  - `env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_train.py --config configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml --device cuda`
- Best checkpoint:
  - `models/exported/pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt`
- Best epoch:
  - `10`
- Best validation balanced accuracy:
  - `0.6750869675397978`

### MaxViT export refresh

- Train export:
  - `env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_eval.py --config configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml --checkpoint models/exported/pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt --split train --device cuda`
- Val export:
  - `env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_eval.py --config configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml --checkpoint models/exported/pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt --split val --device cuda`
- Test export:
  - `env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_eval.py --config configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml --checkpoint models/exported/pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt --split test --device cuda`

### ConvNeXtV2 crop export refresh

- Train / val / test exports rerun with:
  - `env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_eval.py --config configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml --checkpoint models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt --split <train|val|test> --device cuda`

### Ensemble evaluation

- Equal-weight ensemble:
  - `env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/run_late_fusion.py --config configs/inference_pattern_convnext_maxvit_avgprob_eq.yaml --device cuda`
- Validation-tuned ensemble:
  - `env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/run_late_fusion.py --config configs/inference_pattern_convnext_maxvit_avgprob_valtuned.yaml --device cuda`

## Metric Table

Important source note:

- Frozen truth docs store authoritative test metrics for the official single-model benchmark and the deployed late-fusion rule.
- Those frozen test values are kept authoritative below.
- Validation metrics for those two frozen references come from the current local artifacts because the frozen handoff docs did not record validation metrics.

| Line | Val BA | Val Macro F1 | Test BA | Test Macro F1 | Test source |
| --- | ---: | ---: | ---: | ---: | --- |
| frozen official single-model benchmark | `0.6916` | `0.6606` | `0.8482` | `0.7990` | frozen truth |
| frozen deployed late-fusion rule | `0.7128` | `0.6867` | `0.8563` | `0.8115` | frozen truth |
| MaxViT-Tiny single model | `0.6751` | `0.6782` | `0.7976` | `0.8001` | current execution |
| ConvNeXtV2 + MaxViT equal weights | `0.6841` | `0.6822` | `0.8151` | `0.7961` | current execution |
| ConvNeXtV2 + MaxViT validation-tuned | `0.7254` | `0.7118` | `0.8265` | `0.7943` | current execution |

## Validation-Tuned Ensemble Rule

- Config:
  - `configs/inference_pattern_convnext_maxvit_avgprob_valtuned.yaml`
- Search space:
  - scalar probability weight grid, step `0.05`
- Selection split:
  - validation only
- Chosen weights:
  - ConvNeXtV2 crop: `0.60`
  - MaxViT-Tiny crop: `0.40`

## Leakage And Alignment Checks

- Canonical holdout split preserved for every run
- Shared label mapping preserved across ConvNeXtV2, MaxViT-Tiny, and fused outputs
- Prediction export contract enforced:
  - required columns present
  - fixed probability-column order present
  - prediction metadata sidecars present
- Alignment verified by direct comparison of exported tables:
  - train counts `498 / 498`, same order `True`, same targets `True`
  - val counts `106 / 106`, same order `True`, same targets `True`
  - test counts `108 / 108`, same order `True`, same targets `True`
- No test tuning of ensemble weights
- No TG, severity, segmentation, or mask-derived codepaths were used
- No preprocessing was fit on combined train+val+test labels

## Comparison Against Frozen Baselines

### MaxViT-Tiny single model

- vs frozen official single-model benchmark:
  - test BA delta `-0.0506`
  - test macro F1 delta `+0.0011`
- vs frozen deployed late-fusion rule:
  - test BA delta `-0.0587`
  - test macro F1 delta `-0.0114`

### Equal-weight ensemble

- vs frozen official single-model benchmark:
  - test BA delta `-0.0331`
  - test macro F1 delta `-0.0029`
- vs frozen deployed late-fusion rule:
  - test BA delta `-0.0412`
  - test macro F1 delta `-0.0154`

### Validation-tuned ensemble

- vs frozen official single-model benchmark:
  - test BA delta `-0.0217`
  - test macro F1 delta `-0.0047`
- vs frozen deployed late-fusion rule:
  - test BA delta `-0.0299`
  - test macro F1 delta `-0.0172`

## Frozen-Truth Note

Current local reruns under the frozen experiment paths still drift below the authoritative frozen test benchmarks:

- current local official-single rerun:
  - test BA `0.7979`
  - test macro F1 `0.7500`
- current local deployed-rule rerun:
  - test BA `0.7764`
  - test macro F1 `0.7438`

These local reruns were useful for validation exports and alignment checks, but they do not overwrite the frozen truth recorded in `codex.md` and the final modeling handoff docs.

## Verdict

- MaxViT-Tiny is a stronger image-only partner than the earlier Swin-T follow-up in the sense that the validation-tuned ensemble moved closer to the frozen ConvNeXtV2 references.
- MaxViT-Tiny alone is still below the frozen official single-model benchmark on test balanced accuracy.
- The equal-weight ensemble improved over MaxViT alone, but it still missed both frozen baselines.
- The validation-tuned ensemble was the best of the new clean image-only MaxViT combinations:
  - test BA `0.8265`
  - test macro F1 `0.7943`
- Even so, it did not beat:
  - the frozen official single-model benchmark at test BA `0.8482`
  - the frozen deployed late-fusion rule at test BA `0.8563`
- Promotion status:
  - not promotable
  - worth keeping as a clean complementary-model check, but not as a new recommended pattern line
