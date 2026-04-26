import yaml
import os
from pathlib import Path

def main():
    out_dir = Path("configs/cv_pattern_3class/w0035_style")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_config = "../../train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml"
    
    for i in range(10):
        config_path = out_dir / f"fold_{i:02d}.yaml"
        exp_name = f"pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__ordinalaux_w0035style__cv10_fold{i:02d}__seed42"
        
        cfg = {
            "base_config": base_config,
            "experiment_name": exp_name,
            "output_root": "outputs/cv_pattern_3class/w0035_style",
            "epochs": 6,
            "early_stopping_patience": 3,
            "lr": 0.00002,
            "num_workers": 4,
            "ordinal_aux_weight": 0.035,
            "seed": 4200 + i,
            "split_file": f"data/interim/split_files/cv_pattern_3class/fold_{i:02d}.csv",
            "model": {
                "ordinal_aux": {
                    "enabled": True
                }
            }
        }
        
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    main()
