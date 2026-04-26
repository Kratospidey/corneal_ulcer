import os
import json
import csv
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

from config_utils import resolve_config
from data.label_utils import get_task_definition
from model_factory import create_model
from data.dataset import build_dataloaders, build_datasets
from data.split_utils import ensure_task_splits, load_manifest, load_split_dataframe
from data.transforms import build_transforms

class FeatureExtractor:
    def __init__(self, model, layers_to_hook):
        self.model = model
        self.features = {}
        self.hooks = []
        # We look for backbone.stem, backbone.stages, etc.
        for name, module in self.model.named_modules():
            if name in layers_to_hook:
                self.hooks.append(module.register_forward_hook(self.get_hook(name)))
                
    def get_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, (list, tuple)):
                output = output[0]
            if isinstance(output, dict):
                output = output.get("logits", output.get("features", list(output.values())[0]))
            
            # Handle NCHW or NHWC
            if output.ndim == 4:
                # Global average pool
                if output.shape[1] < output.shape[3]: # Likely NHWC
                     self.features[name] = output.mean(dim=(1, 2)).detach().cpu().numpy()
                else: # Likely NCHW
                     self.features[name] = output.mean(dim=(2, 3)).detach().cpu().numpy()
            elif output.ndim == 2:
                self.features[name] = output.detach().cpu().numpy()
            else:
                # Flatten
                self.features[name] = output.reshape(output.shape[0], -1).detach().cpu().numpy()
        return hook

    def close(self):
        for hook in self.hooks:
            hook.remove()

def run_layer_probe():
    RUN_NAME = "pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42"
    checkpoint_path = Path(f"models/exported/{RUN_NAME}/best.pt")
    config_path = Path(f"configs/model_improve/ordinal_weight_grid/train_pattern3_officialinit_ordinalaux_w0035.yaml")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = resolve_config(config_path)
    task_config = resolve_config(config["task_config"])
    split_config = resolve_config(config["split_config"])
    task_def = get_task_definition(task_config["task_name"])
    class_names = task_def.class_names
    
    output_dir = Path(f"outputs/layer_probe/{RUN_NAME}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = create_model(config["model"], num_classes=len(class_names)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Identify layers to hook
    layers_to_hook = []
    layers_to_hook.append("backbone.stem")
    for i in range(4):
        layers_to_hook.append(f"backbone.stages.{i}")
        num_blocks = [3, 3, 9, 3][i]
        for j in range(num_blocks):
            layers_to_hook.append(f"backbone.stages.{i}.blocks.{j}")
    layers_to_hook.append("backbone.norm_pre")
    layers_to_hook.append("classifier")
    
    # Validate layers exist
    valid_layers = []
    for name, _ in model.named_modules():
        if name in layers_to_hook:
            valid_layers.append(name)
            
    print(f"Hooking {len(valid_layers)} layers.")
    
    extractor = FeatureExtractor(model, valid_layers)
    
    # Setup data loading similar to main_eval.py
    split_paths = ensure_task_splits(
        manifest_path=split_config["manifest_path"],
        duplicate_csv_path=split_config["duplicate_candidates_path"],
        split_dir=split_config["split_dir"],
        task_name=task_def.task_name,
        label_column=task_def.label_column,
        holdout_seed=int(split_config.get("holdout", {}).get("seed", 42)),
        cv_seed=int(split_config.get("repeated_cv", {}).get("seed", 42)),
    )
    manifest_df = load_manifest(split_config["manifest_path"])
    split_path = Path(config.get("split_file", split_paths["holdout"]))
    split_df = load_split_dataframe(split_path)
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
        preprocessing_mode=str(config.get("preprocessing_mode", "raw_rgb")),
    )
    dataloaders = build_dataloaders(
        datasets=datasets,
        batch_size=int(config.get("batch_size", 16)),
        num_workers=int(config.get("num_workers", 4)),
        sampler=None,
        shuffle_train=False,
    )
    
    all_features = {split: {name: [] for name in valid_layers} for split in ["train", "val", "test"]}
    all_labels = {split: [] for split in ["train", "val", "test"]}
    
    with torch.no_grad():
        for split in ["train", "val", "test"]:
            for batch in tqdm(dataloaders[split], desc=f"Extracting {split} features"):
                images = batch["image"].to(device)
                labels = batch["target"].cpu().numpy()
                all_labels[split].extend(labels)
                
                model(images)
                for name in valid_layers:
                    all_features[split][name].append(extractor.features[name])
    
    extractor.close()
    
    # Concatenate features
    for split in ["train", "val", "test"]:
        all_labels[split] = np.array(all_labels[split])
        for name in valid_layers:
            all_features[split][name] = np.concatenate(all_features[split][name], axis=0)
            
    # Probe training
    results = []
    seeds = [1, 7, 21]
    Cs = [0.1, 1.0, 10.0]
    
    for name in tqdm(valid_layers, desc="Probing layers"):
        layer_scores = []
        for seed in seeds:
            for C in Cs:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=C, class_weight="balanced", random_state=seed, max_iter=1000))
                ])
                pipe.fit(all_features["train"][name], all_labels["train"])
                val_preds = pipe.predict(all_features["val"][name])
                ba = balanced_accuracy_score(all_labels["val"], val_preds)
                layer_scores.append(ba)
                
        results.append({
            "Layer": valid_layers.index(name) + 1,
            "Layer Name": name,
            "Mean BA": np.mean(layer_scores),
            "Max BA": np.max(layer_scores),
            "Min BA": np.min(layer_scores),
            "Median BA": np.median(layer_scores),
            "Std BA": np.std(layer_scores)
        })
        
    # Sort by Mean BA descending
    results.sort(key=lambda x: x["Mean BA"], reverse=True)
    
    # Save CSV
    with open(output_dir / "layer_probe_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    # Generate Markdown Table
    md_table = "### Table. Balanced accuracy rates of ConvNeXtV2 Tiny w0035 features based on each layer with classification, sorted according to mean balanced accuracy rates.\n\n"
    md_table += "| Layer | Layer Name | Mean BA | Max BA | Min BA | Median BA |\n"
    md_table += "|---|---|---|---|---|---|\n"
    
    for i, res in enumerate(results):
        row = f"| {res['Layer']} | {res['Layer Name']} | {res['Mean BA']:.4f} | {res['Max BA']:.4f} | {res['Min BA']:.4f} | {res['Median BA']:.4f} |"
        if i < 5:
            row = f"| **{res['Layer']}** | **{res['Layer Name']}** | **{res['Mean BA']:.4f}** | **{res['Max BA']:.4f}** | **{res['Min BA']:.4f}** | **{res['Median BA']:.4f}** |"
        md_table += row + "\n"
        
    md_table += "\n*The five highest mean balanced-accuracy layers are bolded.*\n"
    
    with open(output_dir / "layer_probe_top_table.md", "w") as f:
        f.write(md_table)
        
    # Also copy to reports
    reports_dir = Path("outputs/reports/model_improve")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "w0035_layer_probe_top_table.md", "w") as f:
        f.write(md_table)

    # Generate Two-column LaTeX Table
    tex_content = r"\begin{table}[h]" + "\n"
    tex_content += r"\centering" + "\n"
    tex_content += r"\caption{Balanced accuracy rates of ConvNeXtV2 Tiny w0035 features based on each layer with classification, sorted according to mean balanced accuracy rates.}" + "\n"
    tex_content += r"\begin{tabular}{|l|l|l|l|l|l||l|l|l|l|l|l|}" + "\n"
    tex_content += r"\hline" + "\n"
    tex_content += r"L & Layer Name & Mean & Max & Min & Med & L & Layer Name & Mean & Max & Min & Med \\ \hline" + "\n"
    
    mid = (len(results) + 1) // 2
    for i in range(mid):
        r1 = results[i]
        name1 = r1['Layer Name'].replace('_', '\\_')
        row_str = ""
        if i < 5:
            row_str += f"\\textbf{{{r1['Layer']}}} & \\textbf{{{name1}}} & \\textbf{{{r1['Mean BA']:.4f}}} & \\textbf{{{r1['Max BA']:.4f}}} & \\textbf{{{r1['Min BA']:.4f}}} & \\textbf{{{r1['Median BA']:.4f}}}"
        else:
            row_str += f"{r1['Layer']} & {name1} & {r1['Mean BA']:.4f} & {r1['Max BA']:.4f} & {r1['Min BA']:.4f} & {r1['Median BA']:.4f}"
            
        if i + mid < len(results):
            r2 = results[i + mid]
            name2 = r2['Layer Name'].replace('_', '\\_')
            # Top 5 check for second column (only if total layers < 10, but here mid > 5 likely)
            if i + mid < 5:
                 row_str += f" & \\textbf{{{r2['Layer']}}} & \\textbf{{{name2}}} & \\textbf{{{r2['Mean BA']:.4f}}} & \\textbf{{{r2['Max BA']:.4f}}} & \\textbf{{{r2['Min BA']:.4f}}} & \\textbf{{{r2['Median BA']:.4f}}}"
            else:
                 row_str += f" & {r2['Layer']} & {name2} & {r2['Mean BA']:.4f} & {r2['Max BA']:.4f} & {r2['Min BA']:.4f} & {r2['Median BA']:.4f}"
        else:
            row_str += " & & & & & & "
            
        tex_content += row_str + r" \\ \hline" + "\n"
        
    tex_content += r"\end{tabular}" + "\n"
    tex_content += r"\end{table}" + "\n"
    
    with open(output_dir / "layer_probe_balanced_accuracy_table_twocol.tex", "w") as f:
        f.write(tex_content)
        
    # Simple bar chart
    plt.figure(figsize=(12, 6))
    names = [r["Layer Name"] for r in results[:20]]
    means = [r["Mean BA"] for r in results[:20]]
    plt.bar(names, means)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Balanced Accuracy")
    plt.title("Top 20 Layers by Mean Balanced Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "layer_probe_bar_chart.png")
    
    # Manifest
    manifest = {
        "run_name": RUN_NAME,
        "num_layers_probed": len(results),
        "top_layer": results[0]["Layer Name"],
        "top_mean_ba": results[0]["Mean BA"]
    }
    with open(output_dir / "layer_probe_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

if __name__ == "__main__":
    run_layer_probe()
