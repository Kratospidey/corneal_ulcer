import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    
    root_dir = Path(args.root)
    folds = sorted(list(root_dir.glob("fold_*")))
    
    fold_data = []
    recalls_data = []
    
    for fold_dir in folds:
        fold_name = fold_dir.name
        
        val_metrics_path = fold_dir / "val_metrics.json"
        test_metrics_path = fold_dir / "test_metrics.json"
        test_report_path = fold_dir / "test_classification_report.csv"
        
        if not val_metrics_path.exists() or not test_metrics_path.exists() or not test_report_path.exists():
            print(f"Skipping {fold_name} due to missing metric files.")
            continue
            
        with open(val_metrics_path) as f:
            val_metrics = json.load(f)
            
        with open(test_metrics_path) as f:
            test_metrics = json.load(f)
            
        report_df = pd.read_csv(test_report_path, index_col=0)
        pl_recall = report_df.loc["point_like", "recall"] if "point_like" in report_df.index else 0.0
        pfm_recall = report_df.loc["point_flaky_mixed", "recall"] if "point_flaky_mixed" in report_df.index else 0.0
        flaky_recall = report_df.loc["flaky", "recall"] if "flaky" in report_df.index else 0.0
        
        # Try to find best epoch from training_summary.json
        best_epoch = "N/A"
        train_sum_path = fold_dir / "training_summary.json"
        if train_sum_path.exists():
            with open(train_sum_path) as f:
                train_sum = json.load(f)
                best_epoch = train_sum.get("best_epoch", "N/A")
                
        fold_data.append({
            "Fold": fold_name,
            "Val BA": val_metrics.get("balanced_accuracy", 0),
            "Test BA": test_metrics.get("balanced_accuracy", 0),
            "Test Macro F1": test_metrics.get("macro_f1", 0),
            "Test Weighted F1": test_metrics.get("weighted_f1", 0),
            "Accuracy": test_metrics.get("accuracy", 0),
            "PL Recall": pl_recall,
            "PFM Recall": pfm_recall,
            "Flaky Recall": flaky_recall,
            "Best Epoch": best_epoch,
            "Notes": ""
        })
        
        recalls_data.append({
            "Fold": fold_name,
            "point_like": pl_recall,
            "point_flaky_mixed": pfm_recall,
            "flaky": flaky_recall
        })
        
    if not fold_data:
        print("No fold data found to aggregate.")
        return
        
    df = pd.DataFrame(fold_data)
    recalls_df = pd.DataFrame(recalls_data)
    
    # Calculate statistics
    metrics = ["Val BA", "Test BA", "Test Macro F1", "Test Weighted F1", "Accuracy", "PL Recall", "PFM Recall", "Flaky Recall"]
    
    summary_data = []
    for m in metrics:
        vals = df[m].values
        summary_data.append({
            "Metric": m,
            "Mean": np.mean(vals),
            "Std": np.std(vals),
            "Min": np.min(vals),
            "Max": np.max(vals),
            "Median": np.median(vals),
        })
    summary_df = pd.DataFrame(summary_data)
    
    class_summary_data = []
    for c in ["point_like", "point_flaky_mixed", "flaky"]:
        vals = recalls_df[c].values
        class_summary_data.append({
            "Class": c,
            "Recall Mean": np.mean(vals),
            "Recall Std": np.std(vals),
            "Recall Min": np.min(vals),
            "Recall Max": np.max(vals),
        })
    class_summary_df = pd.DataFrame(class_summary_data)
    
    # Save outputs
    out_dir = Path("outputs/reports/cv_pattern_3class/w0035_style")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(out_dir / "CV10_RESULTS.csv", index=False)
    recalls_df.to_csv(out_dir / "CV10_PER_CLASS_RECALL.csv", index=False)
    
    summary_dict = {
        "folds_aggregated": len(df),
        "overall_summary": summary_data,
        "per_class_summary": class_summary_data
    }
    with open(out_dir / "CV10_SUMMARY.json", "w") as f:
        json.dump(summary_dict, f, indent=4)
        
    # Markdown report
    md_lines = [
        "# CV Results (w0035-style recipe)",
        "",
        "The fixed holdout result remains the frozen deployment benchmark. The new cross-validation benchmark estimates the robustness of the w0035-style training recipe across alternative stratified splits.",
        "",
        "## Fixed Holdout Anchors",
        "- official: BA 0.8482, macro F1 0.7990",
        "- w0035: BA 0.8671, macro F1 0.8546",
        "",
        "The CV result is not directly identical to the fixed holdout because each fold trains a fresh model from normal pretrained initialization and evaluates on a different split. It estimates recipe robustness, not the exact w0035 checkpoint performance.",
        "",
        "## Summary Statistics",
        "| Metric | Mean | Std | Min | Max | Median |",
        "|---|---|---|---|---|---|"
    ]
    
    for r in summary_data:
        md_lines.append(f"| {r['Metric']} | {r['Mean']:.4f} | {r['Std']:.4f} | {r['Min']:.4f} | {r['Max']:.4f} | {r['Median']:.4f} |")
        
    md_lines.extend([
        "",
        "## Per-class Recall Summary",
        "| Class | Recall Mean | Recall Std | Recall Min | Recall Max |",
        "|---|---|---|---|---|"
    ])
    
    for r in class_summary_data:
        md_lines.append(f"| {r['Class']} | {r['Recall Mean']:.4f} | {r['Recall Std']:.4f} | {r['Recall Min']:.4f} | {r['Recall Max']:.4f} |")
        
    md_lines.extend([
        "",
        "## Fold-wise Results",
        "| Fold | Val BA | Test BA | Test Macro F1 | Accuracy | PL Recall | PFM Recall | Flaky Recall | Best Epoch | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|"
    ])
    
    for _, r in df.iterrows():
        md_lines.append(f"| {r['Fold']} | {r['Val BA']:.4f} | {r['Test BA']:.4f} | {r['Test Macro F1']:.4f} | {r['Accuracy']:.4f} | {r['PL Recall']:.4f} | {r['PFM Recall']:.4f} | {r['Flaky Recall']:.4f} | {r['Best Epoch']} | {r['Notes']} |")
        
    (out_dir / "CV10_RESULTS.md").write_text("\n".join(md_lines))
    print(f"Generated aggregate reports in {out_dir}")

if __name__ == "__main__":
    main()