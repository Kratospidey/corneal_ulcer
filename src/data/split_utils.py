from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import logging


def load_manifest(manifest_path: str | Path):
    import pandas as pd  # type: ignore

    return pd.read_csv(manifest_path)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
            return value
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def duplicate_group_map(manifest_df, duplicate_csv_path: str | Path) -> dict[str, str]:
    import pandas as pd  # type: ignore

    uf = UnionFind()
    all_ids = [str(value) for value in manifest_df["image_id"].astype(str).tolist()]
    for image_id in all_ids:
        uf.find(image_id)

    duplicate_path = Path(duplicate_csv_path)
    if duplicate_path.exists() and duplicate_path.stat().st_size > 0:
        duplicate_df = pd.read_csv(duplicate_path)
        for _, group_df in duplicate_df.groupby("group_id"):
            image_ids = [str(value) for value in group_df["image_id"].astype(str).tolist()]
            for index in range(1, len(image_ids)):
                uf.union(image_ids[0], image_ids[index])
    return {image_id: uf.find(image_id) for image_id in all_ids}


def _mode_label(values: list[str]) -> str:
    counts = Counter(values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _build_group_records(manifest_df, label_column: str, group_map: dict[str, str]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for _, row in manifest_df.iterrows():
        image_id = str(row["image_id"])
        group_id = group_map[image_id]
        record = records.setdefault(
            group_id,
            {"group_id": group_id, "image_ids": [], "labels": [], "size": 0},
        )
        record["image_ids"].append(image_id)
        record["labels"].append(str(row[label_column]))
        record["size"] += 1
    output: list[dict[str, Any]] = []
    for record in records.values():
        output.append(
            {
                "group_id": record["group_id"],
                "image_ids": record["image_ids"],
                "group_label": _mode_label(record["labels"]),
                "size": record["size"],
            }
        )
    return output


def _train_test_split_groups(group_records: list[dict[str, Any]], train_size: float, seed: int):
    from sklearn.model_selection import train_test_split  # type: ignore

    labels = [record["group_label"] for record in group_records]
    indices = list(range(len(group_records)))
    try:
        train_indices, test_indices = train_test_split(
            indices,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
    except ValueError:
        train_indices, test_indices = train_test_split(
            indices,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    return [group_records[index] for index in train_indices], [group_records[index] for index in test_indices]


def generate_holdout_split(
    manifest_df,
    task_name: str,
    label_column: str,
    split_dir: str | Path,
    duplicate_csv_path: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("Holdout split ratios must sum to 1.0")

    group_map = duplicate_group_map(manifest_df, duplicate_csv_path)
    group_records = _build_group_records(manifest_df, label_column, group_map)
    train_groups, temp_groups = _train_test_split_groups(group_records, train_size=train_ratio, seed=seed)
    temp_train_ratio = val_ratio / (val_ratio + test_ratio)
    val_groups, test_groups = _train_test_split_groups(temp_groups, train_size=temp_train_ratio, seed=seed + 1)

    split_lookup: dict[str, str] = {}
    for split_name, groups in (("train", train_groups), ("val", val_groups), ("test", test_groups)):
        for group in groups:
            for image_id in group["image_ids"]:
                split_lookup[str(image_id)] = split_name

    rows: list[dict[str, Any]] = []
    for _, row in manifest_df.iterrows():
        image_id = str(row["image_id"])
        rows.append(
            {
                "image_id": image_id,
                "group_id": group_map[image_id],
                "task_name": task_name,
                "label": str(row[label_column]),
                "split": split_lookup[image_id],
            }
        )

    split_path = Path(split_dir) / f"{task_name}_holdout.csv"
    write_split_rows(split_path, rows)
    validate_no_overlap(rows)
    validate_group_integrity(rows)
    return rows, split_path


def generate_repeated_cv_split(
    manifest_df,
    task_name: str,
    label_column: str,
    split_dir: str | Path,
    duplicate_csv_path: str | Path,
    n_splits: int = 5,
    n_repeats: int = 3,
    val_fraction_within_train: float = 0.1764705882,
    seed: int = 42,
):
    from sklearn.model_selection import StratifiedKFold  # type: ignore

    group_map = duplicate_group_map(manifest_df, duplicate_csv_path)
    group_records = _build_group_records(manifest_df, label_column, group_map)
    group_labels = [record["group_label"] for record in group_records]
    rows: list[dict[str, Any]] = []

    for repeat in range(n_repeats):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        indices = list(range(len(group_records)))
        for fold, (train_val_idx, test_idx) in enumerate(splitter.split(indices, group_labels)):
            train_val_groups = [group_records[index] for index in train_val_idx]
            test_groups = [group_records[index] for index in test_idx]
            train_groups, val_groups = _train_test_split_groups(
                train_val_groups,
                train_size=1.0 - val_fraction_within_train,
                seed=seed + repeat + fold + 17,
            )
            split_lookup: dict[str, str] = {}
            for split_name, groups in (("train", train_groups), ("val", val_groups), ("test", test_groups)):
                for group in groups:
                    for image_id in group["image_ids"]:
                        split_lookup[str(image_id)] = split_name

            for _, row in manifest_df.iterrows():
                image_id = str(row["image_id"])
                rows.append(
                    {
                        "image_id": image_id,
                        "group_id": group_map[image_id],
                        "task_name": task_name,
                        "label": str(row[label_column]),
                        "repeat": repeat,
                        "fold": fold,
                        "split": split_lookup[image_id],
                    }
                )

    split_path = Path(split_dir) / f"{task_name}_repeated_cv.csv"
    write_split_rows(split_path, rows)
    return rows, split_path


def write_split_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    import csv

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        file_path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def validate_no_overlap(rows: list[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        image_id = str(row["image_id"])
        split = str(row["split"])
        if image_id in seen and seen[image_id] != split:
            raise ValueError(f"Image {image_id} appears in multiple splits")
        seen[image_id] = split


def validate_group_integrity(rows: list[dict[str, Any]]) -> None:
    group_to_split: dict[str, str] = {}
    for row in rows:
        group_id = str(row["group_id"])
        split = str(row["split"])
        if group_id in group_to_split and group_to_split[group_id] != split:
            raise ValueError(f"Duplicate-aware group {group_id} crosses splits")
        group_to_split[group_id] = split


def ensure_task_splits(
    manifest_path: str | Path,
    duplicate_csv_path: str | Path,
    split_dir: str | Path,
    task_name: str,
    label_column: str,
    holdout_seed: int = 42,
    cv_seed: int = 42,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    manifest_df = load_manifest(manifest_path)
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    holdout_path = split_dir / f"{task_name}_holdout.csv"
    cv_path = split_dir / f"{task_name}_repeated_cv.csv"
    if not holdout_path.exists():
        generate_holdout_split(
            manifest_df,
            task_name=task_name,
            label_column=label_column,
            split_dir=split_dir,
            duplicate_csv_path=duplicate_csv_path,
            seed=holdout_seed,
        )
        if logger:
            logger.info("Generated holdout split %s", holdout_path)
    if not cv_path.exists():
        generate_repeated_cv_split(
            manifest_df,
            task_name=task_name,
            label_column=label_column,
            split_dir=split_dir,
            duplicate_csv_path=duplicate_csv_path,
            seed=cv_seed,
        )
        if logger:
            logger.info("Generated repeated CV split %s", cv_path)
    return {"holdout": holdout_path, "repeated_cv": cv_path}


def load_split_dataframe(split_path: str | Path):
    import pandas as pd  # type: ignore

    return pd.read_csv(split_path)


def summarize_split_distribution(split_df) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = split_df.groupby(["split", "label"]).size().reset_index(name="count")
    for _, row in grouped.iterrows():
        rows.append({"split": row["split"], "label": row["label"], "count": int(row["count"])})
    return rows
