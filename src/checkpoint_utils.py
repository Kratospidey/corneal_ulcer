from __future__ import annotations

from pathlib import Path
from typing import Any


def _candidate_target_keys(source_key: str, model_state: dict[str, Any]) -> list[str]:
    candidates: list[str] = [source_key]
    if not source_key.startswith("backbone."):
        candidates.append(f"backbone.{source_key}")
    if source_key.startswith("head.fc."):
        candidates.append(f"classifier.{source_key.removeprefix('head.fc.')}")
    if source_key.startswith("head.norm."):
        suffix = source_key.removeprefix("head.norm.")
        candidates.append(f"feature_norm.{suffix}")
        candidates.append(f"backbone.head.norm.{suffix}")
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in model_state and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def load_model_init_checkpoint(model, checkpoint_path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    import torch  # type: ignore

    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()

    filtered_state = {}
    skipped_mismatches: list[str] = []
    for key, value in state_dict.items():
        for target_key in _candidate_target_keys(key, model_state):
            target = model_state[target_key]
            if tuple(value.shape) != tuple(target.shape):
                skipped_mismatches.append(f"{key}->{target_key}")
                continue
            filtered_state[target_key] = value

    load_result = model.load_state_dict(filtered_state, strict=False)
    return {
        "checkpoint_path": str(checkpoint_file),
        "loaded_keys": len(filtered_state),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "skipped_shape_mismatch_keys": skipped_mismatches,
    }
