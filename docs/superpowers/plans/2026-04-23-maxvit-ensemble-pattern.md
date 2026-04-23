# MaxViT Ensemble Pattern Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real `MaxViT-Tiny` pattern-3 single-model line plus one equal-weight and one validation-tuned probability-space ensemble against the frozen ConvNeXtV2 crop line, then report honest val/test results.

**Architecture:** Extend the existing pattern-only `timm` training stack with one MaxViT config, enforce a shared prediction-export contract through a small reusable helper, and reuse `src/run_late_fusion.py` for the two-model probability ensemble with hard alignment checks. Keep the scope image-only and stop at the planned small validation grid even if MaxViT is weak.

**Tech Stack:** Python 3.11, PyTorch, timm, torchvision, scikit-learn, YAML configs, unittest

---

## File Map

### Create

- `configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml`
- `configs/inference_pattern_convnext_maxvit_avgprob_eq.yaml`
- `configs/inference_pattern_convnext_maxvit_avgprob_valtuned.yaml`
- `src/evaluation/prediction_contract.py`
- `tests/test_prediction_contract.py`
- `docs/superpowers/handoffs/MAXVIT_ENSEMBLE_PATTERN_PLAN.md`
- `docs/superpowers/handoffs/MAXVIT_ENSEMBLE_PATTERN_RESULTS.md`

### Modify

- `src/model_factory.py`
- `src/evaluation/evaluate.py`
- `src/evaluation/reports.py`
- `src/run_late_fusion.py`
- `tests/test_pattern_only_tasks.py`
- `README_training.md`
- `codex.md`

### Outputs Expected From Execution

- `models/checkpoints/pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt`
- `models/exported/pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt`
- `outputs/predictions/pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/{train,val,test}_predictions.csv`
- `outputs/predictions/pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_eq__holdout_v1__seed42/{val,test}_predictions.csv`
- `outputs/predictions/pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_valtuned__holdout_v1__seed42/{val,test}_predictions.csv`
- `outputs/metrics/.../val_metrics.json`
- `outputs/metrics/.../test_metrics.json`

### Runtime Interpreter

Use the direct environment interpreter to avoid the stale `micromamba` proc lock:

- `/home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python`

All commands below assume `env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python ...`

### Task 1: Add MaxViT Model Family Support

**Files:**
- Modify: `src/model_factory.py`
- Test: `tests/test_pattern_only_tasks.py`

- [ ] **Step 1: Write the failing test for MaxViT model creation**

```python
def test_maxvit_tiny_model_is_supported(self) -> None:
    model = create_model(
        {"name": "maxvit_tiny_tf_224.in1k", "pretrained": False, "freeze_backbone": False},
        num_classes=3,
    )
    self.assertIsNotNone(model)
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python -m unittest tests.test_pattern_only_tasks -v
```

Expected: FAIL with `ValueError: Unsupported model: maxvit_tiny_tf_224.in1k`

- [ ] **Step 3: Implement minimal MaxViT support in the factory**

```python
elif model_name.startswith(("convnextv2", "swin", "maxvit")):
    import timm
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, ...)
```

Also extend `freeze_feature_extractor` so `maxvit*` models freeze everything except the classifier head when requested.

- [ ] **Step 4: Re-run the targeted test**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python -m unittest tests.test_pattern_only_tasks -v
```

Expected: PASS for the new MaxViT support test

- [ ] **Step 5: Commit the model-family support**

```bash
git add src/model_factory.py tests/test_pattern_only_tasks.py
git commit -m "feat: add MaxViT support to pattern model factory"
```

### Task 2: Add The MaxViT Training Config

**Files:**
- Create: `configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml`
- Modify: `tests/test_pattern_only_tasks.py`

- [ ] **Step 1: Write the failing config-resolution test**

```python
def test_maxvit_pattern_config_resolves_expected_overrides(self) -> None:
    config = resolve_config("configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml")
    self.assertEqual(config["preprocessing_mode"], "cornea_crop_scale_v1")
    self.assertEqual(config["train_transform_profile"], "pattern_augplus_v2")
    self.assertEqual(config["sampler"], "weighted_sampler_tempered")
    self.assertEqual(config["sampler_temperature"], 0.65)
    self.assertEqual(config["model"]["name"], "maxvit_tiny_tf_224.in1k")
```

- [ ] **Step 2: Run the targeted test to verify it fails on the missing config**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python -m unittest tests.test_pattern_only_tasks -v
```

Expected: FAIL with missing-config file error

- [ ] **Step 3: Add the MaxViT training config**

```yaml
base_config: train_convnextv2_raw.yaml
experiment_name: pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42
preprocessing_mode: cornea_crop_scale_v1
train_transform_profile: pattern_augplus_v2
sampler: weighted_sampler_tempered
sampler_temperature: 0.65
promotion_reference_config: configs/pattern_promotion_references.yaml
model:
  name: maxvit_tiny_tf_224.in1k
  pretrained: true
  freeze_backbone: false
  drop_path_rate: 0.1
```

- [ ] **Step 4: Re-run the targeted test**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python -m unittest tests.test_pattern_only_tasks -v
```

Expected: PASS with the new config resolving correctly

- [ ] **Step 5: Commit the config**

```bash
git add configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml tests/test_pattern_only_tasks.py
git commit -m "feat: add MaxViT pattern training config"
```

### Task 3: Enforce The Prediction Export Contract

**Files:**
- Create: `src/evaluation/prediction_contract.py`
- Modify: `src/evaluation/evaluate.py`
- Modify: `src/evaluation/reports.py`
- Create: `tests/test_prediction_contract.py`

- [ ] **Step 1: Write the failing contract tests**

```python
def test_prediction_row_contains_required_columns(self) -> None:
    row = build_prediction_row(
        base_row={"image_id": "5", "split": "val", "target_index": 0, "predicted_index": 1},
        class_names=("point_like", "point_flaky_mixed", "flaky"),
        probabilities=[0.1, 0.7, 0.2],
    )
    self.assertEqual(
        list(row.keys())[:4],
        ["image_id", "split", "target_index", "predicted_index"],
    )
```

```python
def test_probability_column_order_is_fixed(self) -> None:
    self.assertEqual(
        probability_column_names(("point_like", "point_flaky_mixed", "flaky")),
        ["prob_point_like", "prob_point_flaky_mixed", "prob_flaky"],
    )
```

- [ ] **Step 2: Run the contract tests to verify they fail**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python -m unittest tests.test_prediction_contract -v
```

Expected: FAIL because `prediction_contract.py` does not exist yet

- [ ] **Step 3: Implement the reusable export-contract helper**

```python
REQUIRED_BASE_COLUMNS = ("image_id", "split", "target_index", "predicted_index")

def probability_column_names(class_names):
    return [f"prob_{name}" for name in class_names]

def build_prediction_row(base_row, class_names, probabilities):
    ...

def build_prediction_provenance(task_name, class_names, split_name, source_config_path, checkpoint_path=None):
    ...
```

The helper should:
- build canonical row ordering
- expose class-order provenance
- provide schema validation helpers reusable by fusion

- [ ] **Step 4: Switch inference/export code to the canonical names**

Update `src/evaluation/evaluate.py` so the inference payload emits `predicted_index` as the canonical field. Keep `pred_index` only if a compatibility alias is needed internally, but all saved exports must contain `predicted_index`.

Update `src/evaluation/reports.py` so saved prediction CSVs and any sidecar metadata/provenance file are generated through `prediction_contract.py`.

- [ ] **Step 5: Re-run the new tests**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python -m unittest tests.test_prediction_contract -v
```

Expected: PASS with stable canonical column order

- [ ] **Step 6: Run the task-only regression test**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python -m unittest tests.test_pattern_only_tasks -v
```

Expected: PASS with no pattern-only config regressions

- [ ] **Step 7: Commit the export-contract work**

```bash
git add src/evaluation/prediction_contract.py src/evaluation/evaluate.py src/evaluation/reports.py tests/test_prediction_contract.py
git commit -m "feat: enforce prediction export contract"
```

### Task 4: Add Hard Alignment Validation And Ensemble Configs

**Files:**
- Modify: `src/run_late_fusion.py`
- Create: `configs/inference_pattern_convnext_maxvit_avgprob_eq.yaml`
- Create: `configs/inference_pattern_convnext_maxvit_avgprob_valtuned.yaml`
- Modify: `tests/test_pattern_only_tasks.py`
- Modify: `tests/test_prediction_contract.py`

- [ ] **Step 1: Write the failing alignment tests**

```python
def test_fusion_rejects_target_index_mismatch(self) -> None:
    with self.assertRaises(ValueError):
        validate_component_tables(component_tables=[left_table, right_table], class_names=CLASS_NAMES)
```

```python
def test_fusion_rejects_probability_schema_mismatch(self) -> None:
    with self.assertRaises(ValueError):
        validate_component_rows(...)
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python -m unittest tests.test_prediction_contract -v
```

Expected: FAIL because the fusion validator does not exist yet

- [ ] **Step 3: Factor `run_late_fusion.py` around explicit validation helpers**

```python
def validate_component_tables(component_tables, class_names):
    ...

def _require_same_sample_coverage(component_tables):
    ...

def _require_same_target_indices(component_tables):
    ...
```

The validation must fail hard on:
- missing required columns
- probability-column name/order mismatch
- class-order provenance mismatch
- duplicate `image_id`
- sample coverage mismatch
- `target_index` mismatch for the same `image_id`

- [ ] **Step 4: Add the two MaxViT ensemble configs**

Equal-weight config:
```yaml
experiment_name: pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_eq__holdout_v1__seed42
inference:
  normalization_mode: probability
  decision_rule: weighted_average
  weight_search:
    fixed_weights: [0.5, 0.5]
```

Val-tuned config:
```yaml
experiment_name: pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_valtuned__holdout_v1__seed42
inference:
  normalization_mode: probability
  decision_rule: weighted_average
  weight_search:
    grid_step: 0.05
```

Both configs should point at:
- the frozen ConvNeXtV2 crop training config and checkpoint
- the new MaxViT training config and checkpoint

- [ ] **Step 5: Add config-resolution coverage for the new ensemble configs**

Extend `tests/test_pattern_only_tasks.py` to assert:
- task config resolves to `pattern_3class`
- both ensemble configs contain exactly 2 models
- the MaxViT component points at the new config path

- [ ] **Step 6: Re-run the targeted tests**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python -m unittest tests.test_prediction_contract tests.test_pattern_only_tasks -v
```

Expected: PASS with schema and config validation covered

- [ ] **Step 7: Commit the fusion-guard work**

```bash
git add src/run_late_fusion.py configs/inference_pattern_convnext_maxvit_avgprob_eq.yaml configs/inference_pattern_convnext_maxvit_avgprob_valtuned.yaml tests/test_pattern_only_tasks.py tests/test_prediction_contract.py
git commit -m "feat: add MaxViT fusion configs and alignment guards"
```

### Task 5: Write The Working Handoff Note Before Running Experiments

**Files:**
- Create: `docs/superpowers/handoffs/MAXVIT_ENSEMBLE_PATTERN_PLAN.md`

- [ ] **Step 1: Draft the pipeline-audit handoff**

Include:
- canonical split path
- label order
- shared crop/augmentation/sampler recipe
- prediction contract
- fairness/leakage checks to enforce
- exact experiment names to run

- [ ] **Step 2: Save the handoff doc**

Expected sections:
```md
# MaxViT Ensemble Pattern Plan
## Scope
## Live Pattern Pipeline Assumptions
## Planned Runs
## Leakage Checks
```

- [ ] **Step 3: Commit the handoff doc**

```bash
git add docs/superpowers/handoffs/MAXVIT_ENSEMBLE_PATTERN_PLAN.md
git commit -m "docs: add MaxViT ensemble execution handoff"
```

### Task 6: Execute The MaxViT Single-Model Run

**Files:**
- Uses: `configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml`
- Produces: `models/checkpoints/...`, `models/exported/...`, `outputs/metrics/...`, `outputs/predictions/...`

- [ ] **Step 1: Run the MaxViT training command**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_train.py --config configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml --device cuda
```

Expected: training completes and writes `best.pt`, `training_summary.json`, `val_metrics.json`, and `test_metrics.json`

- [ ] **Step 2: Evaluate and export the train split**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_eval.py --config configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml --checkpoint models/exported/pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt --split train --device cuda
```

Expected: `outputs/predictions/.../train_predictions.csv` exists with the canonical contract

- [ ] **Step 3: Evaluate and export the val split**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_eval.py --config configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml --checkpoint models/exported/pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt --split val --device cuda
```

Expected: `val_predictions.csv` and `val_metrics.json` exist

- [ ] **Step 4: Evaluate and export the test split**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_eval.py --config configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml --checkpoint models/exported/pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt --split test --device cuda
```

Expected: `test_predictions.csv`, `test_metrics.json`, confusion matrix, and summary report exist

- [ ] **Step 5: Record the single-model sanity outcome**

Inspect:
- `outputs/metrics/pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/val_metrics.json`
- `outputs/metrics/pattern3__maxvit_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/test_metrics.json`

If MaxViT is clearly weak, continue with only the required equal-weight and one val-tuned ensemble. Do not add extra variants.

### Task 7: Verify Or Regenerate ConvNeXtV2 Crop Exports

**Files:**
- Uses: `configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml`
- Uses: `models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt`

- [ ] **Step 1: Inspect existing ConvNeXt exports for the required schema**

Check:
- `outputs/predictions/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/train_predictions.csv`
- `outputs/predictions/.../val_predictions.csv`
- `outputs/predictions/.../test_predictions.csv`

Expected: all three exist and contain `image_id`, `split`, `target_index`, `predicted_index`, and the fixed probability columns

- [ ] **Step 2: If any file is missing or has the wrong schema, regenerate train/val/test**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/main_eval.py --config configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml --checkpoint models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/best.pt --split train --device cuda
```

Repeat for `val` and `test`.

- [ ] **Step 3: Verify the frozen-truth note**

If the current local ConvNeXt artifact under the frozen experiment path re-evaluates below the frozen benchmark, preserve that discrepancy in the final report instead of overwriting the frozen official numbers.

### Task 8: Run The Two Required Ensemble Evaluations

**Files:**
- Uses: `configs/inference_pattern_convnext_maxvit_avgprob_eq.yaml`
- Uses: `configs/inference_pattern_convnext_maxvit_avgprob_valtuned.yaml`

- [ ] **Step 1: Run the equal-weight ensemble**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/run_late_fusion.py --config configs/inference_pattern_convnext_maxvit_avgprob_eq.yaml --device cuda
```

Expected: val/test metrics, reports, confusion matrices, and predictions are written under `pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_eq__holdout_v1__seed42`

- [ ] **Step 2: Run the validation-tuned ensemble**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python src/run_late_fusion.py --config configs/inference_pattern_convnext_maxvit_avgprob_valtuned.yaml --device cuda
```

Expected: val/test metrics and `late_fusion_metadata.json` contain the selected validation-only weights

- [ ] **Step 3: Verify the tuning discipline**

Inspect:
- `outputs/reports/pattern3__convnextv2_tiny_plus_maxvit_tiny__avgprob_valtuned__holdout_v1__seed42/late_fusion_metadata.json`

Expected:
- exactly two weights
- weight selected from the `0.05` grid
- no test-driven search

- [ ] **Step 4: Stop the ensemble surface here**

Do not add:
- more ensemble members
- finer-grained grids
- per-class routing
- logit-space follow-up in this pass

### Task 9: Write The Results Handoff And Minimal Live-Doc Updates

**Files:**
- Create: `docs/superpowers/handoffs/MAXVIT_ENSEMBLE_PATTERN_RESULTS.md`
- Modify: `README_training.md`
- Modify: `codex.md`

- [ ] **Step 1: Build the five-line comparison table from real artifacts**

Collect:
- frozen official single-model benchmark
- frozen deployed late-fusion rule
- MaxViT single model
- equal-weight ensemble
- val-tuned ensemble

Required metrics:
- val balanced accuracy
- val macro F1
- test balanced accuracy
- test macro F1
- delta vs frozen official single
- delta vs frozen deployed rule

- [ ] **Step 2: Write the results handoff**

The doc must include:
```md
# MaxViT Ensemble Pattern Results
## Scope
## Live Pipeline Assumptions Used
## Runs Actually Executed
## Metric Table
## Validation-Tuned Ensemble Rule
## Leakage And Alignment Checks
## Comparison Against Frozen Baselines
## Verdict
```

- [ ] **Step 3: Update `README_training.md` only if the run path is real and reproducible**

Add the exact MaxViT train command and both ensemble commands beside the existing pattern commands.

- [ ] **Step 4: Update `codex.md` only if the results are real**

Add:
- the new MaxViT single-model artifact
- the two new ensemble artifacts
- a short honest verdict on whether they beat the frozen references

- [ ] **Step 5: Commit the docs update**

```bash
git add docs/superpowers/handoffs/MAXVIT_ENSEMBLE_PATTERN_RESULTS.md README_training.md codex.md
git commit -m "docs: record MaxViT ensemble pattern results"
```

### Task 10: Final Verification Before Hand-Off

**Files:**
- Check only generated artifacts and docs from Tasks 6-9

- [ ] **Step 1: Re-run the focused test suite**

Run:
```bash
env PYTHONPATH=src /home/kratospidey/.local/share/mamba/envs/corneal-train/bin/python -m unittest tests.test_pattern_only_tasks tests.test_prediction_contract tests.test_promotion -v
```

Expected: PASS

- [ ] **Step 2: Verify the required artifact set exists**

Check for:
- MaxViT `best.pt`
- MaxViT `train/val/test` predictions
- equal-weight ensemble `val/test` metrics
- val-tuned ensemble `val/test` metrics
- val-tuned `late_fusion_metadata.json`
- both new handoff docs

- [ ] **Step 3: Verify the final user-facing answer can be exact**

Confirm you can report:
- exact experiment names
- exact metric values
- exact selected weights
- exact commands run
- exact files changed

- [ ] **Step 4: Hand off without blurring frozen truth**

The final answer must separate:
- frozen official single-model benchmark
- frozen deployed inference rule
- current local reruns
- new MaxViT single-model result
- new ensemble results
