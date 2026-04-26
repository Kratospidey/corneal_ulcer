import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="pattern_3class")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest-path", default="data/interim/manifests/manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser

def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.manifest_path)
    label_col = f"task_{args.task}"
    df = df.dropna(subset=[label_col])
    
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    
    splits = list(skf.split(df, df[label_col]))
    
    summaries = []
    
    for fold_idx in range(args.n_splits):
        test_idx = splits[fold_idx][1]
        val_idx = splits[(fold_idx + 1) % args.n_splits][1]
        
        test_set = set(test_idx)
        val_set = set(val_idx)
        train_idx = [i for i in range(len(df)) if i not in test_set and i not in val_set]
        
        split_df = pd.DataFrame({"image_id": df["image_id"]})
        split_df["split"] = ""
        split_df.loc[train_idx, "split"] = "train"
        split_df.loc[val_idx, "split"] = "val"
        split_df.loc[test_idx, "split"] = "test"
        
        fold_csv = out_dir / f"fold_{fold_idx:02d}.csv"
        split_df.to_csv(fold_csv, index=False)
        
        train_counts = df.iloc[train_idx][label_col].value_counts().to_dict()
        val_counts = df.iloc[val_idx][label_col].value_counts().to_dict()
        test_counts = df.iloc[test_idx][label_col].value_counts().to_dict()
        
        total_train = len(train_idx)
        total_val = len(val_idx)
        total_test = len(test_idx)
        
        summaries.append({
            "fold": fold_idx,
            "train_count": total_train,
            "val_count": total_val,
            "test_count": total_test,
            "train_counts": train_counts,
            "val_counts": val_counts,
            "test_counts": test_counts,
        })
        
    total_counts = df[label_col].value_counts().to_dict()
    
    md_lines = [
        f"# CV Split Summary for {args.task}",
        "",
        f"**Total Dataset Count:** {len(df)}",
        f"**Total Class Counts:** {total_counts}",
        f"**Splitter:** StratifiedKFold (n_splits={args.n_splits}, seed={args.seed})",
        "",
        "| Fold | Train | Val | Test | Train Breakdown | Val Breakdown | Test Breakdown |",
        "|---|---|---|---|---|---|---|"
    ]
    
    csv_rows = []
    for s in summaries:
        train_bd = ", ".join([f"{k}: {v} ({v/s['train_count']*100:.1f}%)" for k,v in s["train_counts"].items()])
        val_bd = ", ".join([f"{k}: {v} ({v/s['val_count']*100:.1f}%)" for k,v in s["val_counts"].items()])
        test_bd = ", ".join([f"{k}: {v} ({v/s['test_count']*100:.1f}%)" for k,v in s["test_counts"].items()])
        
        md_lines.append(f"| {s['fold']} | {s['train_count']} | {s['val_count']} | {s['test_count']} | {train_bd} | {val_bd} | {test_bd} |")
        
        csv_rows.append({
            "fold": s["fold"],
            "train_count": s["train_count"],
            "val_count": s["val_count"],
            "test_count": s["test_count"],
            "train_breakdown": train_bd,
            "val_breakdown": val_bd,
            "test_breakdown": test_bd,
        })
        
    (out_dir / "cv_split_summary.md").write_text("\n".join(md_lines))
    pd.DataFrame(csv_rows).to_csv(out_dir / "cv_split_summary.csv", index=False)
    print(f"Generated {args.n_splits} folds in {out_dir}")

if __name__ == "__main__":
    main()