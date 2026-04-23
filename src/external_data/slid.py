from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

import pandas as pd
from torch.utils.data import DataLoader, Dataset  # type: ignore

from utils_io import safe_open_image


def ensure_slid_images_extracted(
    zip_path: str | Path,
    extracted_dir: str | Path,
    logger=None,
) -> Path:
    zip_path = Path(zip_path)
    extracted_dir = Path(extracted_dir)
    sentinel = extracted_dir / "1.png"
    if sentinel.exists():
        return extracted_dir
    if not zip_path.exists():
        raise FileNotFoundError(f"SLID image zip not found: {zip_path}")
    extracted_dir.parent.mkdir(parents=True, exist_ok=True)
    if logger is not None:
        logger.info("Extracting SLID images from %s to %s", zip_path, extracted_dir.parent)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extracted_dir.parent)
    if not sentinel.exists():
        raise FileNotFoundError(f"Expected extracted SLID image missing after extraction: {sentinel}")
    return extracted_dir


def load_slid_manifest(manifest_path: str | Path):
    manifest_df = pd.read_csv(manifest_path)
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    return manifest_df


def load_slid_split(split_path: str | Path):
    split_df = pd.read_csv(split_path)
    split_df["image_id"] = split_df["image_id"].astype(str)
    return split_df


def resolve_slid_manifest_paths(
    manifest_df,
    extracted_dir: str | Path,
):
    manifest_df = manifest_df.copy()
    extracted_dir = Path(extracted_dir)
    manifest_df["raw_image_path"] = manifest_df["image_filename"].map(lambda name: str(extracted_dir / str(name)))
    return manifest_df


def build_slid_rows(manifest_df, split_df, split_name: str):
    split_rows = split_df[split_df["split"] == split_name][["image_id"]].copy()
    split_rows["image_id"] = split_rows["image_id"].astype(str)
    manifest_df = manifest_df.copy()
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    merged = manifest_df.merge(split_rows, on="image_id", how="inner")
    return merged.loc[merged["has_cornea_mask"].astype(str).str.lower() == "true"].reset_index(drop=True)


class SlidCorneaSegmentationDataset(Dataset):
    def __init__(self, rows, split_name: str, transform=None) -> None:
        self.rows = rows.reset_index(drop=True)
        self.split_name = split_name
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        image = safe_open_image(Path(str(row["raw_image_path"]))).convert("RGB")
        cornea_mask = safe_open_image(Path(str(row["cornea_mask_path"]))).convert("L")

        sample = {
            "image": image,
            "masks": {
                "cornea": cornea_mask,
                "ulcer": None,
            },
            "target_available": {
                "cornea": True,
                "ulcer": False,
            },
        }
        if self.transform is not None:
            sample = self.transform(sample)

        return {
            "image": sample["image"],
            "targets": sample["targets"],
            "target_available": sample["target_available"],
            "image_id": str(row["image_id"]),
            "raw_image_path": str(row["raw_image_path"]),
            "cornea_mask_path": str(row["cornea_mask_path"]),
            "ulcer_mask_path": "",
            "split": self.split_name,
        }


def build_slid_cornea_datasets(manifest_df, split_df, transforms_by_split: dict[str, object]):
    datasets: dict[str, SlidCorneaSegmentationDataset] = {}
    for split_name in ("train", "val", "test"):
        rows = build_slid_rows(manifest_df, split_df, split_name)
        datasets[split_name] = SlidCorneaSegmentationDataset(
            rows=rows,
            split_name=split_name,
            transform=transforms_by_split[split_name],
        )
    return datasets


def build_slid_cornea_dataloaders(
    datasets: dict[str, SlidCorneaSegmentationDataset],
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }


def slid_split_summary(datasets: dict[str, SlidCorneaSegmentationDataset]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split_name, dataset in datasets.items():
        summary[split_name] = {
            "rows": int(len(dataset.rows)),
            "cornea_supervised_rows": int(len(dataset.rows)),
            "ulcer_supervised_rows": 0,
        }
    return summary
