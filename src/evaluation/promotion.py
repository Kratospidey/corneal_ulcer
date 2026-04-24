from __future__ import annotations

from pathlib import Path
from typing import Any

from config_utils import resolve_config


def resolve_promotion_config(config_value: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if config_value is None:
        return {}
    if isinstance(config_value, (str, Path)):
        return resolve_config(config_value)
    return dict(config_value)


def build_promotion_reference_block(
    metrics: dict[str, Any],
    task_name: str,
    promotion_config: str | Path | dict[str, Any] | None,
) -> dict[str, Any]:
    config = resolve_promotion_config(promotion_config)
    if not config:
        return {}

    references = dict(config.get("references", {}))
    official = dict(references.get("official_single", {}))
    deployed = dict(references.get("deployed_rule", {}))
    if official.get("task_name") and str(official["task_name"]) != str(task_name):
        return {}

    balanced_accuracy = metrics.get("balanced_accuracy")
    macro_f1 = metrics.get("macro_f1")

    return {
        "promotion_reference": {
            "official_single": official,
            "deployed_rule": deployed,
        },
        "delta_vs_official_single_ba": _delta(balanced_accuracy, official.get("balanced_accuracy")),
        "delta_vs_deployed_rule_ba": _delta(balanced_accuracy, deployed.get("balanced_accuracy")),
        "delta_vs_official_single_macro_f1": _delta(macro_f1, official.get("macro_f1")),
        "delta_vs_deployed_rule_macro_f1": _delta(macro_f1, deployed.get("macro_f1")),
    }


def _delta(current_value: Any, reference_value: Any) -> float | None:
    if current_value is None or reference_value is None:
        return None
    return float(current_value) - float(reference_value)
