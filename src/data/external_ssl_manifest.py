import os
from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image
from tqdm import tqdm
import scipy.io

def fast_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filepath, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()

def get_slid_data(base_dir: Path):
    slid_dir = base_dir / "SLID"
    records = []
    if slid_dir.exists():
        for file in slid_dir.rglob("*.*"):
            if "__MACOSX" in file.parts:
                continue
            if file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                continue
            records.append({
                "source_dataset": "SLID",
                "source_subset": "Original_Slit-lamp_Images",
                "image_path": str(file.resolve()),
                "filename": file.name,
                "extension": file.suffix.lower(),
                "modality": "unknown"
            })
    return records

def get_slitnet_data(base_dir: Path):
    slitnet_dir = base_dir / "SLIT Net"
    records = []
    for subset in ["Blue_Light", "White_Light"]:
        subset_dir = slitnet_dir / subset
        if subset_dir.exists():
            for file in subset_dir.rglob("*.*"):
                if "__MACOSX" in file.parts:
                    continue
                if file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    continue
                records.append({
                    "source_dataset": "SLIT-Net",
                    "source_subset": subset,
                    "image_path": str(file.resolve()),
                    "filename": file.name,
                    "extension": file.suffix.lower(),
                    "modality": "blue_light" if subset == "Blue_Light" else "white_light"
                })
    return records

def main():
    base_dir = Path("External Datasets")
    out_dir = Path("data/external_manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    records = get_slid_data(base_dir) + get_slitnet_data(base_dir)
    print(f"Found {len(records)} potential images.")
    
    processed_records = []
    bad_files = []
    seen_hashes = {}
    duplicates = []
    
    for i, r in enumerate(tqdm(records)):
        path = Path(r["image_path"])
        try:
            with Image.open(path) as img:
                r["width"], r["height"] = img.size
                r["mode"] = img.mode
                r["readable"] = True
                
            file_hash = fast_hash(path)
            r["sha256_or_fast_hash"] = file_hash
            r["patient_or_case_id"] = "unknown"  # No obvious case id mapping
            
            if file_hash in seen_hashes:
                r["duplicate_of"] = seen_hashes[file_hash]
                duplicates.append(r)
                continue
                
            seen_hashes[file_hash] = r["image_path"]
            r["image_id"] = f"ext_{i:06d}"
            
            # Deterministic split by hash
            # ~90% train, ~10% val
            hash_val = int(file_hash[:8], 16)
            r["split"] = "ssl_val" if (hash_val % 100) < 10 else "ssl_train"
            
            processed_records.append(r)
            
        except Exception as e:
            r["readable"] = False
            r["error"] = str(e)
            bad_files.append(r)
            
    df = pd.DataFrame(processed_records)
    df.to_csv(out_dir / "slitlamp_external_ssl_manifest.csv", index=False)
    
    if bad_files:
        pd.DataFrame(bad_files).to_csv(out_dir / "slitlamp_external_ssl_bad_files.csv", index=False)
    if duplicates:
        pd.DataFrame(duplicates).to_csv(out_dir / "slitlamp_external_ssl_duplicates.csv", index=False)
        
    # Summary
    summary = [
        "# External SSL Manifest Summary",
        f"- Total images processed: {len(df)}",
        f"- Unreadable files: {len(bad_files)}",
        f"- Duplicates removed: {len(duplicates)}",
        f"- ssl_train count: {sum(df['split'] == 'ssl_train')}",
        f"- ssl_val count: {sum(df['split'] == 'ssl_val')}",
        "",
        "## Dataset Counts",
    ]
    
    for ds, count in df.groupby("source_dataset").size().items():
        summary.append(f"- {ds}: {count}")
        
    summary.append("\n## Modality Counts")
    for mod, count in df.groupby("modality").size().items():
        summary.append(f"- {mod}: {count}")
        
    summary.append("\n## Examples")
    for _, row in df.head(3).iterrows():
        summary.append(f"- {row['image_id']}: {row['image_path']}")
        
    with open(out_dir / "slitlamp_external_ssl_manifest_summary.md", "w") as f:
        f.write("\n".join(summary))
        
    print(f"Saved manifest to {out_dir / 'slitlamp_external_ssl_manifest.csv'}")

if __name__ == "__main__":
    main()
