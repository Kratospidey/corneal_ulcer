import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data.external_ssl_transforms import build_ssl_train_transforms
from data.external_ssl_dataset import build_ssl_dataloaders

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    img = tensor.numpy() * std + mean
    img = np.clip(img, 0, 1)
    return img

def main():
    manifest_path = "data/external_manifests/slitlamp_external_ssl_manifest.csv"
    out_dir = Path("outputs/reports/external_ssl/pretrain_aug_preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_transform = build_ssl_train_transforms()
    loaders = build_ssl_dataloaders(manifest_path, train_transform, None, batch_size=4, num_workers=0)
    train_loader = loaders["train"]
    
    # We want examples from SLID, SLIT-Net Blue_Light, SLIT-Net White_Light
    # So we'll iterate until we find at least one of each
    
    found = {"SLID": False, "blue_light": False, "white_light": False}
    saved_images = []
    
    for batch in train_loader:
        images = batch["image"]
        modalities = batch["modality"]
        datasets = batch["dataset"]
        image_ids = batch["image_id"]
        
        for i in range(len(image_ids)):
            ds = datasets[i]
            mod = modalities[i]
            
            key = "SLID" if ds == "SLID" else mod
            
            if not found.get(key, True):
                found[key] = True
                
                view1 = denormalize(images[0][i:i+1])[0].transpose(1, 2, 0)
                view2 = denormalize(images[1][i:i+1])[0].transpose(1, 2, 0)
                
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(view1)
                axes[0].set_title("View 1")
                axes[0].axis('off')
                
                axes[1].imshow(view2)
                axes[1].set_title("View 2")
                axes[1].axis('off')
                
                plt.suptitle(f"Dataset: {ds} | Modality: {mod}")
                save_path = out_dir / f"preview_{key}.png"
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
                
                saved_images.append({
                    "key": key,
                    "dataset": ds,
                    "modality": mod,
                    "path": f"pretrain_aug_preview/preview_{key}.png"
                })
                
        if all(found.values()):
            break
            
    md_lines = ["# External SSL Pretraining Augmentation Preview", ""]
    for img in saved_images:
        md_lines.append(f"## {img['dataset']} - {img['modality']}")
        md_lines.append(f"![{img['key']}]({img['path']})")
        md_lines.append("")
        
    md_path = Path("outputs/reports/external_ssl/pretrain_aug_preview.md")
    md_path.write_text("\n".join(md_lines))
    print(f"Generated preview report at {md_path}")

if __name__ == "__main__":
    main()
