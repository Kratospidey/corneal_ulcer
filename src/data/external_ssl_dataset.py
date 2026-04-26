import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ExternalSSLDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, split: str = "ssl_train", transform=None):
        """
        Dataset for self-supervised learning on external slit-lamp images.
        """
        self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Fallback to a zero image if unreadable (should be caught by manifest generation, but just in case)
            print(f"Warning: Failed to load {image_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform is not None:
            image = self.transform(image)
            
        return {
            "image": image,
            "image_id": row["image_id"],
            "modality": row["modality"],
            "dataset": row["source_dataset"]
        }

def build_ssl_dataloaders(manifest_path: str, train_transform, eval_transform, batch_size: int = 32, num_workers: int = 4):
    df = pd.read_csv(manifest_path)
    
    train_dataset = ExternalSSLDataset(df, split="ssl_train", transform=train_transform)
    val_dataset = ExternalSSLDataset(df, split="ssl_val", transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return {"train": train_loader, "val": val_loader}
