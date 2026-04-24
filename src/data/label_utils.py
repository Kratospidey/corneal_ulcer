from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskDefinition:
    task_name: str
    label_column: str
    class_names: tuple[str, ...]
    description: str
    recommended_metric: str = "balanced_accuracy"


ACTIVE_TASK_DEFINITIONS: dict[str, TaskDefinition] = {
    "pattern_3class": TaskDefinition(
        task_name="pattern_3class",
        label_column="task_pattern_3class",
        class_names=("point_like", "point_flaky_mixed", "flaky"),
        description="Primary Stage 3 task grounded in the dataset spreadsheet Category column.",
    ),
}

ARCHIVED_TASK_DEFINITIONS: dict[str, TaskDefinition] = {
    "severity_5class": TaskDefinition(
        task_name="severity_5class",
        label_column="task_severity_5class",
        class_names=("no_ulcer", "ulcer_leq_25pct", "ulcer_leq_50pct", "ulcer_geq_75pct", "central_ulcer"),
        description="Archived historical severity task; not supported by active entrypoints.",
    ),
    "binary": TaskDefinition(
        task_name="binary",
        label_column="task_binary",
        class_names=("no_ulcer", "ulcer_present"),
        description="Archived historical binary task; not supported by active entrypoints.",
    ),
}

TASK_DEFINITIONS: dict[str, TaskDefinition] = {**ACTIVE_TASK_DEFINITIONS, **ARCHIVED_TASK_DEFINITIONS}


def get_task_definition(task_name: str, allow_archived: bool = False) -> TaskDefinition:
    if task_name in ACTIVE_TASK_DEFINITIONS:
        return ACTIVE_TASK_DEFINITIONS[task_name]
    if task_name in ARCHIVED_TASK_DEFINITIONS:
        if allow_archived:
            return ARCHIVED_TASK_DEFINITIONS[task_name]
        raise KeyError(
            f"Task '{task_name}' is archived. Active training/eval/inference entrypoints only support 'pattern_3class'."
        )
    raise KeyError(f"Unsupported task: {task_name}")


def class_to_index(class_names: list[str] | tuple[str, ...]) -> dict[str, int]:
    return {name: index for index, name in enumerate(class_names)}


def index_to_class(class_names: list[str] | tuple[str, ...]) -> dict[int, str]:
    return {index: name for index, name in enumerate(class_names)}


def label_counts(rows: list[dict[str, Any]], label_column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row[label_column])
        counts[label] = counts.get(label, 0) + 1
    return counts
