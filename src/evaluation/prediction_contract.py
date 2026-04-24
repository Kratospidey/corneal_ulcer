from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_BASE_COLUMNS = ("image_id", "split", "target_index", "predicted_index")


def probability_column_names(class_names: list[str] | tuple[str, ...]) -> list[str]:
    return [f"prob_{class_name}" for class_name in class_names]


def logit_column_names(class_names: list[str] | tuple[str, ...]) -> list[str]:
    return [f"logit_{class_name}" for class_name in class_names]


def build_prediction_provenance(
    task_name: str,
    class_names: list[str] | tuple[str, ...],
    split_name: str,
    source_config_path: str | Path | None,
    checkpoint_path: str | Path | None = None,
    include_logits: bool = False,
) -> dict[str, Any]:
    payload = {
        "task_name": str(task_name),
        "split": str(split_name),
        "class_names": [str(class_name) for class_name in class_names],
        "probability_columns": probability_column_names(class_names),
        "source_config_path": str(source_config_path) if source_config_path else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }
    if include_logits:
        payload["logit_columns"] = logit_column_names(class_names)
    return payload


def build_prediction_row(
    base_row: dict[str, Any],
    class_names: list[str] | tuple[str, ...],
    probabilities: list[float] | tuple[float, ...],
    logits: list[float] | tuple[float, ...] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "image_id": str(base_row["image_id"]),
        "split": str(base_row["split"]),
        "target_index": int(base_row["target_index"]),
        "predicted_index": int(base_row["predicted_index"]),
    }
    for class_name, probability in zip(class_names, probabilities, strict=True):
        row[f"prob_{class_name}"] = float(probability)
    if logits is not None:
        for class_name, logit in zip(class_names, logits, strict=True):
            row[f"logit_{class_name}"] = float(logit)
    if extras:
        for key, value in extras.items():
            if key in row:
                continue
            row[key] = value
    return row


def validate_prediction_rows(
    rows: list[dict[str, Any]],
    class_names: list[str] | tuple[str, ...],
    split_name: str | None = None,
) -> None:
    if not rows:
        raise ValueError("Prediction export rows are empty.")

    expected_probability_columns = probability_column_names(class_names)
    required_columns = set(REQUIRED_BASE_COLUMNS).union(expected_probability_columns)
    expected_logit_columns = logit_column_names(class_names)
    seen_image_ids: set[str] = set()

    for row in rows:
        missing_columns = sorted(required_columns.difference(row.keys()))
        if missing_columns:
            raise ValueError(f"Prediction export is missing required columns: {missing_columns}")

        actual_probability_columns = [key for key in row.keys() if key.startswith("prob_")]
        if actual_probability_columns != expected_probability_columns:
            raise ValueError(
                "Prediction export probability columns do not match the fixed class order: "
                f"expected {expected_probability_columns}, got {actual_probability_columns}"
            )
        actual_logit_columns = [key for key in row.keys() if key.startswith("logit_")]
        if actual_logit_columns and actual_logit_columns != expected_logit_columns:
            raise ValueError(
                "Prediction export logit columns do not match the fixed class order: "
                f"expected {expected_logit_columns}, got {actual_logit_columns}"
            )

        image_id = str(row["image_id"])
        if image_id in seen_image_ids:
            raise ValueError(f"Prediction export contains duplicate image_id '{image_id}'.")
        seen_image_ids.add(image_id)

        if split_name is not None and str(row["split"]) != str(split_name):
            raise ValueError(
                f"Prediction export split mismatch for image_id '{image_id}': "
                f"expected '{split_name}', got '{row['split']}'"
            )


def validate_prediction_provenance(
    provenance: dict[str, Any],
    task_name: str,
    class_names: list[str] | tuple[str, ...],
    split_name: str,
) -> None:
    expected_class_names = [str(class_name) for class_name in class_names]
    expected_probability_columns = probability_column_names(class_names)
    actual_class_names = [str(class_name) for class_name in provenance.get("class_names", [])]
    actual_probability_columns = [str(column) for column in provenance.get("probability_columns", [])]
    actual_logit_columns = [str(column) for column in provenance.get("logit_columns", [])]

    if str(provenance.get("task_name")) != str(task_name):
        raise ValueError(
            f"Prediction provenance task mismatch: expected '{task_name}', got '{provenance.get('task_name')}'"
        )
    if str(provenance.get("split")) != str(split_name):
        raise ValueError(
            f"Prediction provenance split mismatch: expected '{split_name}', got '{provenance.get('split')}'"
        )
    if actual_class_names != expected_class_names:
        raise ValueError(
            "Prediction provenance class order mismatch: "
            f"expected {expected_class_names}, got {actual_class_names}"
        )
    if actual_probability_columns != expected_probability_columns:
        raise ValueError(
            "Prediction provenance probability columns mismatch: "
            f"expected {expected_probability_columns}, got {actual_probability_columns}"
        )
    if actual_logit_columns and actual_logit_columns != logit_column_names(class_names):
        raise ValueError(
            "Prediction provenance logit columns mismatch: "
            f"expected {logit_column_names(class_names)}, got {actual_logit_columns}"
        )
