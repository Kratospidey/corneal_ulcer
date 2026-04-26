import argparse
import subprocess
from pathlib import Path
import sys
import os
import yaml
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--folds", default="all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    config_dir = Path(args.config_dir)
    all_configs = sorted(config_dir.glob("fold_*.yaml"))
    
    if args.folds != "all":
        fold_nums = [int(x) for x in args.folds.split(",")]
        configs_to_run = [c for c in all_configs if int(c.stem.split("_")[1]) in fold_nums]
    else:
        configs_to_run = all_configs
        
    for config_path in configs_to_run:
        fold_name = config_path.stem
        print(f"\n{'='*50}\nRunning CV fold: {fold_name}\n{'='*50}")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        if args.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
            
        train_cmd = [sys.executable, "src/main_train.py", "--config", str(config_path)]
        res = subprocess.run(train_cmd, env=env)
        if res.returncode != 0:
            print(f"Training failed for {fold_name}")
            continue
            
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            
        exp_name = cfg["experiment_name"]
        out_root = Path(cfg.get("output_root", "outputs"))
        ckpt_path = f"models/exported/{exp_name}/best.pt"
        
        if not Path(ckpt_path).exists():
            print(f"Warning: Expected checkpoint not found at {ckpt_path}")
            ckpt_path = f"models/checkpoints/{exp_name}/best.pt"
            
        print(f"Evaluating {fold_name} on test split")
        eval_cmd = [sys.executable, "src/main_eval.py", "--config", str(config_path), "--checkpoint", ckpt_path, "--split", "test", "--device", args.device]
        res = subprocess.run(eval_cmd, env=env)
        if res.returncode != 0:
            print(f"Evaluation failed for {fold_name}")
            
        # Organize outputs into fold_XX folders
        fold_out_dir = out_root / fold_name
        fold_out_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(config_path, fold_out_dir / "config.yaml")
        
        metrics_dir = out_root / "metrics" / exp_name
        reports_dir = out_root / "reports" / exp_name
        predictions_dir = out_root / "predictions" / exp_name
        
        if metrics_dir.exists():
            for f in metrics_dir.glob("*.json"):
                shutil.copy(f, fold_out_dir / f.name)
            for f in metrics_dir.glob("*.csv"):
                shutil.copy(f, fold_out_dir / f.name)
        if reports_dir.exists():
            for f in reports_dir.glob("*.md"):
                shutil.copy(f, fold_out_dir / f.name)
        if predictions_dir.exists():
            for f in predictions_dir.glob("*.csv"):
                shutil.copy(f, fold_out_dir / f.name)
                
        with open(fold_out_dir / "best_checkpoint_path.txt", "w") as f:
            f.write(str(ckpt_path) + "\n")

if __name__ == "__main__":
    main()
