from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset  # type: ignore

from data.label_utils import class_to_index
from preprocessing.paper_preprocessing import load_cornea_mask, prepare_paper_image
from preprocessing.raw_preprocessing import prepare_raw_image
from utils_io import safe_open_image


class CornealUlcerDataset(Dataset):
    def __init__(
        self,
        rows,
        label_column: str,
        class_names: list[str] | tuple[str, ...],
        split_name: str,
        transform=None,
        preprocessing_mode: str = "raw_rgb",
        include_masks: bool = False,
        input_mode: str = "single_crop",
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.label_column = label_column
        self.class_names = tuple(class_names)
        self.label_to_index = class_to_index(self.class_names)
        self.split_name = split_name
        self.transform = transform
        self.preprocessing_mode = preprocessing_mode
        self.include_masks = include_masks
        self.input_mode = input_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        raw_image = safe_open_image(Path(str(row["raw_image_path"])))
        
        if self.input_mode == "dual_crop_context_v1":
            cornea_mask = load_cornea_mask(row.get("cornea_mask_path"))
            image_tight = prepare_paper_image(raw_image.copy(), preprocessing_mode="cornea_crop_scale_v1", cornea_mask=cornea_mask)
            image_wide = prepare_paper_image(raw_image.copy(), preprocessing_mode="cornea_crop_wide_context_v1", cornea_mask=cornea_mask)
            if self.transform is not None:
                image_tight = self.transform(image_tight)
                image_wide = self.transform(image_wide)
            image = image_tight  # for backwards compatibility
        else:
            if self.preprocessing_mode == "raw_rgb":
                image = prepare_raw_image(raw_image)
            else:
                cornea_mask = load_cornea_mask(row.get("cornea_mask_path"))
                image = prepare_paper_image(raw_image, preprocessing_mode=self.preprocessing_mode, cornea_mask=cornea_mask)
            if self.transform is not None:
                image = self.transform(image)

        label_name = str(row[self.label_column])
        payload: dict[str, Any] = {
            "image": image,
            "target": self.label_to_index[label_name],
            "label_name": label_name,
            "image_id": str(row["image_id"]),
            "raw_image_path": str(row["raw_image_path"]),
            "cornea_mask_path": str(row.get("cornea_mask_path", "")),
            "ulcer_mask_path": str(row.get("ulcer_mask_path", "")),
            "split": self.split_name,
        }
        if self.input_mode == "dual_crop_context_v1":
            payload["image_tight"] = image_tight
            payload["image_wide"] = image_wide
            
        return payload

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self.rows[self.label_column].astype(str).tolist():
            counts[label] = counts.get(label, 0) + 1
        return counts


def build_split_rows(manifest_df, split_df, split_name: str):
    split_rows = split_df[split_df["split"] == split_name][["image_id"]]
    split_rows["image_id"] = split_rows["image_id"].astype(str)
    manifest_df = manifest_df.copy()
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    return manifest_df.merge(split_rows, on="image_id", how="inner")


def build_datasets(
    manifest_df,
    split_df,
    label_column: str,
    class_names: list[str] | tuple[str, ...],
    transforms_by_split: dict[str, object],
    preprocessing_mode: str,
    include_masks: bool = False,
    input_mode: str = "single_crop",
) -> dict[str, CornealUlcerDataset]:
    datasets: dict[str, CornealUlcerDataset] = {}
    for split_name in ("train", "val", "test"):
        rows = build_split_rows(manifest_df, split_df, split_name)
        datasets[split_name] = CornealUlcerDataset(
            rows=rows,
            label_column=label_column,
            class_names=class_names,
            split_name=split_name,
            transform=transforms_by_split[split_name],
            preprocessing_mode=preprocessing_mode,
            include_masks=include_masks,
            input_mode=input_mode,
        )
    return datasets


def build_dataloaders(
    datasets: dict[str, CornealUlcerDataset],
    batch_size: int,
    num_workers: int,
    sampler=None,
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=bool(shuffle_train and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders
