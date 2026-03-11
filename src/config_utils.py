from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import json


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    import yaml  # type: ignore

    file_path = Path(path)
    payload = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    base_config_path = config.get("base_config")
    if base_config_path:
        base = resolve_config(Path(config_path).parent / str(base_config_path))
        config = deep_merge(base, config)
    return config


def write_json(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
