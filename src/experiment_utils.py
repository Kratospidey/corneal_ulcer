from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import logging
import os
import random


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("baseline")


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def build_experiment_name(config: dict[str, Any]) -> str:
    if config.get("experiment_name"):
        return str(config["experiment_name"])
    task_name = str(config.get("task_name", "task"))
    model_name = str(config.get("model", {}).get("name", "model"))
    preprocessing_mode = str(config.get("preprocessing_mode", "raw_rgb"))
    return f"{task_name}__{model_name}__{preprocessing_mode}"


def prepare_output_dirs(experiment_name: str, output_root: str | Path = "outputs") -> dict[str, Path]:
    output_root = Path(output_root)
    directories = {
        "experiment_root": output_root / "experiments" / experiment_name,
        "metrics": output_root / "metrics" / experiment_name,
        "reports": output_root / "reports" / experiment_name,
        "confusion_matrices": output_root / "confusion_matrices" / experiment_name,
        "roc_curves": output_root / "roc_curves" / experiment_name,
        "pr_curves": output_root / "pr_curves" / experiment_name,
        "predictions": output_root / "predictions" / experiment_name,
        "explainability": output_root / "explainability" / experiment_name,
        "logs": output_root / "logs" / experiment_name,
        "models": Path("models") / "checkpoints" / experiment_name,
        "exported": Path("models") / "exported" / experiment_name,
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
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


def resolve_device(requested: str = "auto") -> str:
    if requested in {"cpu", "cuda"}:
        return requested
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
