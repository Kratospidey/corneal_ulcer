import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import numpy as np
import copy

import torch

from config_utils import resolve_config
from data.dataset import build_dataloaders, build_datasets
from data.label_utils import get_task_definition
from data.split_utils import ensure_task_splits, load_manifest, load_split_dataframe
from data.transforms import build_transforms
from evaluation.evaluate import run_inference
from evaluation.metrics import compute_classification_metrics
from model_factory import create_model
from checkpoint_utils import load_checkpoint_payload, extract_checkpoint_state_dict

def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate checkpoint soup.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser

def average_checkpoints(checkpoints):
    avg_state_dict = copy.deepcopy(checkpoints[0])
    for key in avg_state_dict.keys():
        for i in range(1, len(checkpoints)):
            avg_state_dict[key] += checkpoints[i][key]
        avg_state_dict[key] = torch.div(avg_state_dict[key], len(checkpoints))
    return avg_state_dict

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    
    config = resolve_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    
    task_config = resolve_config(config["task_config"])
    split_config = resolve_config(config["split_config"])
    task_def = get_task_definition(task_config["task_name"])
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base model
    model = create_model(config["model"], num_classes=len(task_def.class_names)).to(device)
    
    # Load checkpoints
    ckpt_paths = {
        "w0035_seed42": "models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42/best.pt",
        "w0035_seed1": "models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed1/best.pt",
        "w0035_seed7": "models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed7/best.pt",
        "w0035_seed21": "models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed21/best.pt",
        "w0035_seed123": "models/exported/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed123/best.pt"
    }
    
    state_dicts = {}
    for name, path in ckpt_paths.items():
        payload = load_checkpoint_payload(path, map_location=device)
        state_dicts[name] = extract_checkpoint_state_dict(payload)
        
    soups = {
        "Uniform_Top3_Val": average_checkpoints([state_dicts["w0035_seed21"], state_dicts["w0035_seed123"], state_dicts["w0035_seed42"]]),
        "Uniform_All5": average_checkpoints(list(state_dicts.values()))
    }
    
    manifest_df = load_manifest(split_config["manifest_path"])
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
    
    datasets = build_datasets(
        manifest_df=manifest_df,
        split_df=split_df,
        label_column=task_def.label_column,
        class_names=task_def.class_names,
        transforms_by_split=transforms_by_split,
        preprocessing_mode="cornea_crop_scale_v1",
        include_masks=False,
    )
    loaders = build_dataloaders(
        datasets=datasets,
        batch_size=int(config.get("batch_size", 16)),
        num_workers=int(config.get("num_workers", 4)),
        sampler=None,
        shuffle_train=False,
    )
    
    report_rows = []
    
    for soup_name, soup_state in soups.items():
        print(f"Evaluating soup: {soup_name}")
        model.load_state_dict(soup_state)
        model.eval()
        
        soup_metrics = {}
        for split in ["val", "test"]:
            evaluation_payload = run_inference(
                model=model,
                dataloader=loaders[split],
                device=device,
                criterion=None,
                show_progress=True,
            )
            
            metrics = compute_classification_metrics(
                evaluation_payload["y_true"],
                evaluation_payload["y_pred"],
                evaluation_payload["probabilities"],
                task_def.class_names,
            )["metrics"]
            
            soup_metrics[split] = metrics
            
            y_true = np.array(evaluation_payload["y_true"])
            y_pred = np.array(evaluation_payload["y_pred"])
            recalls = {}
            for i, c in enumerate(task_def.class_names):
                mask = y_true == i
                recalls[c] = np.mean(y_pred[mask] == i)
            soup_metrics[f"{split}_recalls"] = recalls
            
        report_rows.append({
            "soup_name": soup_name,
            "included_checkpoints": "seed21, seed123, seed42" if "Top3" in soup_name else "all 5 seeds",
            "val_BA": f"{soup_metrics['val']['balanced_accuracy']:.4f}",
            "val_macro_F1": f"{soup_metrics['val']['macro_f1']:.4f}",
            "test_BA": f"{soup_metrics['test']['balanced_accuracy']:.4f}",
            "test_macro_F1": f"{soup_metrics['test']['macro_f1']:.4f}",
            "accuracy": f"{soup_metrics['test']['accuracy']:.4f}",
            "PL_recall": f"{soup_metrics['test_recalls']['point_like']:.4f}",
            "PFM_recall": f"{soup_metrics['test_recalls']['point_flaky_mixed']:.4f}",
            "flaky_recall": f"{soup_metrics['test_recalls']['flaky']:.4f}",
            "beats_w0035": "Yes" if soup_metrics['test']['balanced_accuracy'] > 0.8671 and soup_metrics['test']['macro_f1'] >= 0.8546 else "No",
            "notes": ""
        })
        
    md_lines = ["# Checkpoint Soup Results", "", "| soup_name | included_checkpoints | val_BA | val_macro_F1 | test_BA | test_macro_F1 | accuracy | PL_recall | PFM_recall | flaky_recall | beats_w0035 | notes |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    
    # Sort by val_BA descending
    report_rows.sort(key=lambda x: float(x["val_BA"]), reverse=True)
    
    for r in report_rows:
        md_lines.append(f"| {r['soup_name']} | {r['included_checkpoints']} | {r['val_BA']} | {r['val_macro_F1']} | {r['test_BA']} | {r['test_macro_F1']} | {r['accuracy']} | {r['PL_recall']} | {r['PFM_recall']} | {r['flaky_recall']} | {r['beats_w0035']} | {r['notes']} |")
        
    out_md = output_dir / "CHECKPOINT_SOUP_RESULTS.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Generated report at {out_md}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())