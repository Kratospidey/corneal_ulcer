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


TASK_DEFINITIONS: dict[str, TaskDefinition] = {
    "pattern_3class": TaskDefinition(
        task_name="pattern_3class",
        label_column="task_pattern_3class",
        class_names=("point_like", "point_flaky_mixed", "flaky"),
        description="Primary Stage 3 task grounded in the dataset spreadsheet Category column.",
    ),
}


def get_task_definition(task_name: str, allow_archived: bool = False) -> TaskDefinition:
    del allow_archived
    if task_name in TASK_DEFINITIONS:
        return TASK_DEFINITIONS[task_name]
    raise KeyError("Active training/eval/inference entrypoints only support 'pattern_3class'.")


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
