import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import numpy as np

import torch

from config_utils import resolve_config
from data.dataset import build_dataloaders, build_datasets
from data.label_utils import get_task_definition
from data.split_utils import ensure_task_splits, load_manifest, load_split_dataframe
from data.transforms import build_transforms
from evaluation.evaluate import run_inference
from evaluation.metrics import compute_classification_metrics
from model_factory import create_model

def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate frozen model using multiple deterministic crop variants.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    
    config = resolve_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    
    task_config = resolve_config(config["task_config"])
    split_config = resolve_config(config["split_config"])
    task_def = get_task_definition(task_config["task_name"])
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = create_model(config["model"], num_classes=len(task_def.class_names)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    manifest_df = load_manifest(split_config["manifest_path"])
    split_path = Path(config.get("split_file", split_paths["holdout"] if "split_paths" in locals() else "data/splits.yaml")) # Fallback if split_paths missing in simplified logic
    
    split_paths = ensure_task_splits(
        manifest_path=split_config["manifest_path"],
        duplicate_csv_path=split_config["duplicate_candidates_path"],
        split_dir=split_config["split_dir"],
        task_name=task_def.task_name,
        label_column=task_def.label_column,
        holdout_seed=int(split_config.get("holdout", {}).get("seed", 42)),
        cv_seed=int(split_config.get("repeated_cv", {}).get("seed", 42)),
    )
    split_df = load_split_dataframe(split_paths["holdout"])

    transforms_by_split = build_transforms(
        int(config.get("image_size", 224)),
        train_profile=str(config.get("train_transform_profile", "default")),
    )
    
    variants = [
        "cornea_crop_scale_v1",
        "cornea_crop_slightly_tight",
        "cornea_crop_slightly_wide",
        "cornea_crop_wide_context_v1",
        "shift_left_up",
        "shift_right_down",
        "shift_left_down",
        "shift_right_up"
    ]
    
    results = {split: {} for split in ["val", "test"]}
    
    for split in ["val", "test"]:
        for variant in variants:
            print(f"Running inference for {split} split, variant {variant}...")
            datasets = build_datasets(
                manifest_df=manifest_df,
                split_df=split_df,
                label_column=task_def.label_column,
                class_names=task_def.class_names,
                transforms_by_split=transforms_by_split,
                preprocessing_mode=variant,
                include_masks=False,
            )
            loaders = build_dataloaders(
                datasets=datasets,
                batch_size=int(config.get("batch_size", 16)),
                num_workers=int(config.get("num_workers", 4)),
                sampler=None,
                shuffle_train=False,
            )
            
            evaluation_payload = run_inference(
                model=model,
                dataloader=loaders[split],
                device=device,
                criterion=None,
                show_progress=True,
            )
            
            results[split][variant] = {
                "y_true": evaluation_payload["y_true"],
                "y_pred": evaluation_payload["y_pred"],
                "probabilities": evaluation_payload["probabilities"],
                "logits": evaluation_payload["logits"],
            }
            
            metrics_payload = compute_classification_metrics(
                evaluation_payload["y_true"],
                evaluation_payload["y_pred"],
                evaluation_payload["probabilities"],
                task_def.class_names,
            )
            print(f"{variant} {split} BA: {metrics_payload['metrics']['balanced_accuracy']:.4f}")

    # View sets
    view_sets = {
        "Set A (base, slightly_wide, slightly_tight)": ["cornea_crop_scale_v1", "cornea_crop_slightly_wide", "cornea_crop_slightly_tight"],
        "Set B (base, wide_context)": ["cornea_crop_scale_v1", "cornea_crop_wide_context_v1"],
        "Set C (base, slightly_wide, shift_left_up, shift_right_down)": ["cornea_crop_scale_v1", "cornea_crop_slightly_wide", "shift_left_up", "shift_right_down"]
    }
    
    report_rows = []
    
    for set_name, views in view_sets.items():
        for split in ["val", "test"]:
            # Logit average
            stacked_logits = np.stack([results[split][v]["logits"] for v in views], axis=0)
            avg_logits = np.mean(stacked_logits, axis=0)
            avg_probs = torch.softmax(torch.tensor(avg_logits), dim=1).numpy()
            avg_preds = np.argmax(avg_probs, axis=1)
            
            y_true = results[split][views[0]]["y_true"]
            
            metrics = compute_classification_metrics(y_true, avg_preds, avg_probs, task_def.class_names)["metrics"]
            
            if split == "val":
                val_ba = metrics["balanced_accuracy"]
                val_f1 = metrics["macro_f1"]
            else:
                test_ba = metrics["balanced_accuracy"]
                test_f1 = metrics["macro_f1"]
                test_acc = metrics["accuracy"]
                # Calculate recalls
                recalls = {}
                for i, c in enumerate(task_def.class_names):
                    mask = np.array(y_true) == i
                    recalls[c] = np.mean(np.array(avg_preds)[mask] == i)
                
                report_rows.append({
                    "run": "w0035",
                    "view_set": set_name,
                    "aggregation": "logit_average",
                    "val_BA": f"{val_ba:.4f}",
                    "val_macro_F1": f"{val_f1:.4f}",
                    "test_BA": f"{test_ba:.4f}",
                    "test_macro_F1": f"{test_f1:.4f}",
                    "accuracy": f"{test_acc:.4f}",
                    "PL_recall": f"{recalls['point_like']:.4f}",
                    "PFM_recall": f"{recalls['point_flaky_mixed']:.4f}",
                    "flaky_recall": f"{recalls['flaky']:.4f}",
                    "beats_w0035": "Yes" if test_ba > 0.8671 and test_f1 >= 0.8546 else "No",
                    "notes": ""
                })

            # Probability average
            stacked_probs = np.stack([results[split][v]["probabilities"] for v in views], axis=0)
            avg_probs_prob = np.mean(stacked_probs, axis=0)
            avg_preds_prob = np.argmax(avg_probs_prob, axis=1)
            
            metrics_prob = compute_classification_metrics(y_true, avg_preds_prob, avg_probs_prob, task_def.class_names)["metrics"]
            
            if split == "val":
                val_ba_prob = metrics_prob["balanced_accuracy"]
                val_f1_prob = metrics_prob["macro_f1"]
            else:
                test_ba_prob = metrics_prob["balanced_accuracy"]
                test_f1_prob = metrics_prob["macro_f1"]
                test_acc_prob = metrics_prob["accuracy"]
                
                recalls_prob = {}
                for i, c in enumerate(task_def.class_names):
                    mask = np.array(y_true) == i
                    recalls_prob[c] = np.mean(np.array(avg_preds_prob)[mask] == i)
                
                report_rows.append({
                    "run": "w0035",
                    "view_set": set_name,
                    "aggregation": "prob_average",
                    "val_BA": f"{val_ba_prob:.4f}",
                    "val_macro_F1": f"{val_f1_prob:.4f}",
                    "test_BA": f"{test_ba_prob:.4f}",
                    "test_macro_F1": f"{test_f1_prob:.4f}",
                    "accuracy": f"{test_acc_prob:.4f}",
                    "PL_recall": f"{recalls_prob['point_like']:.4f}",
                    "PFM_recall": f"{recalls_prob['point_flaky_mixed']:.4f}",
                    "flaky_recall": f"{recalls_prob['flaky']:.4f}",
                    "beats_w0035": "Yes" if test_ba_prob > 0.8671 and test_f1_prob >= 0.8546 else "No",
                    "notes": ""
                })
                
    md_lines = ["# Crop Consistency Results", "", "| run | view_set | aggregation | val_BA | val_macro_F1 | test_BA | test_macro_F1 | accuracy | PL_recall | PFM_recall | flaky_recall | beats_w0035 | notes |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    
    # Sort by val_BA descending
    report_rows.sort(key=lambda x: float(x["val_BA"]), reverse=True)
    
    for r in report_rows:
        md_lines.append(f"| {r['run']} | {r['view_set']} | {r['aggregation']} | {r['val_BA']} | {r['val_macro_F1']} | {r['test_BA']} | {r['test_macro_F1']} | {r['accuracy']} | {r['PL_recall']} | {r['PFM_recall']} | {r['flaky_recall']} | {r['beats_w0035']} | {r['notes']} |")
        
    out_md = output_dir / "CROP_CONSISTENCY_RESULTS.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Generated report at {out_md}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
