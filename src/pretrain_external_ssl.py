import argparse
import json
import os
from pathlib import Path
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from tqdm import tqdm
import yaml

from data.external_ssl_dataset import build_ssl_dataloaders
from data.external_ssl_transforms import build_ssl_train_transforms, build_ssl_eval_transforms

class SimSiam(nn.Module):
    def __init__(self, encoder, dim=2048, pred_dim=512):
        super().__init__()
        self.encoder = encoder
        
        # Build a 3-layer projector
        prev_dim = encoder.num_features
        
        self.projector = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x1, x2):
        f1, f2 = self.encoder(x1), self.encoder(x2)
        z1, z2 = self.projector(f1), self.projector(f2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()

def D(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Train SSL", leave=False):
        x1, x2 = batch["image"]
        x1, x2 = x1.to(device), x2.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            p1, p2, z1, z2 = model(x1, x2)
            loss = D(p1, z2) / 2 + D(p2, z1) / 2
            
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    for batch in tqdm(loader, desc="Eval SSL", leave=False):
        x1, x2 = batch["image"]
        x1, x2 = x1.to(device), x2.to(device)
        
        with torch.amp.autocast('cuda'):
            p1, p2, z1, z2 = model(x1, x2)
            loss = D(p1, z2) / 2 + D(p2, z1) / 2
            
        total_loss += loss.item()
    return total_loss / len(loader)

def hash_file(filepath: Path):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filepath, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config["seed"])
    
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
        
    train_transform = build_ssl_train_transforms(config["image_size"])
    eval_transform = build_ssl_train_transforms(config["image_size"]) # evaluate using augmented views for SSL loss
    
    loaders = build_ssl_dataloaders(
        config["manifest_path"],
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )
    
    encoder = timm.create_model(config["model"]["name"], pretrained=config["model"]["pretrained"], num_classes=0)
    model = SimSiam(encoder).to(device)
    
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    epochs = 1 if args.smoke_test else config["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    scaler = torch.amp.GradScaler('cuda', enabled=config["amp"]) if device == "cuda" and config["amp"] else None
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, loaders["train"], optimizer, scaler, device)
        val_loss = eval_epoch(model, loaders["val"], device)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
    # Save model
    backbone_path = out_dir / "backbone.pt"
    ssl_checkpoint_path = out_dir / "ssl_checkpoint.pt"
    
    torch.save(model.encoder.state_dict(), backbone_path)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history
    }, ssl_checkpoint_path)
    
    with open(out_dir / "pretrain_metrics.json", "w") as f:
        json.dump(history, f, indent=2)
        
    with open(out_dir / "SHA256SUMS.txt", "w") as f:
        f.write(f"{hash_file(backbone_path)}  backbone.pt\n")
        f.write(f"{hash_file(ssl_checkpoint_path)}  ssl_checkpoint.pt\n")
        
    # Generate report
    report_dir = Path("outputs/reports/external_ssl")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# External SSL Pretraining (SimSiam V1)",
        "",
        f"- Config: `{args.config}`",
        f"- Epochs completed: {epochs}",
        f"- Train images: {len(loaders['train'].dataset)}",
        f"- Val images: {len(loaders['val'].dataset)}",
        f"- Final Train Loss: {history['train_loss'][-1]:.4f}",
        f"- Final Val Loss: {history['val_loss'][-1]:.4f}",
        "",
        "## Saved Artifacts",
        f"- `{backbone_path}`",
        f"- `{ssl_checkpoint_path}`"
    ]
    
    report_path = report_dir / "SSL_PRETRAIN_SIMSIAM_V1.md"
    report_path.write_text("\n".join(report_lines))
    print(f"Saved report to {report_path}")

if __name__ == "__main__":
    main()
