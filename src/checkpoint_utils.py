from __future__ import annotations

from pathlib import Path
from typing import Any


def _candidate_target_keys(source_key: str, model_state: dict[str, Any]) -> list[str]:
    candidates: list[str] = [source_key]
    if not source_key.startswith("backbone."):
        candidates.append(f"backbone.{source_key}")
    if source_key.startswith("backbone."):
        candidates.append(source_key.removeprefix("backbone."))
    if source_key.startswith("head.fc."):
        candidates.append(f"classifier.{source_key.removeprefix('head.fc.')}")
    if source_key.startswith("classifier."):
        candidates.append(f"head.fc.{source_key.removeprefix('classifier.')}")
    if source_key.startswith("head.norm."):
        suffix = source_key.removeprefix("head.norm.")
        candidates.append(f"feature_norm.{suffix}")
        candidates.append(f"backbone.head.norm.{suffix}")
    if source_key.startswith("feature_norm."):
        candidates.append(f"head.norm.{source_key.removeprefix('feature_norm.')}")
    if source_key.startswith("backbone.head.norm."):
        candidates.append(f"head.norm.{source_key.removeprefix('backbone.head.norm.')}")
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in model_state and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def load_checkpoint_payload(checkpoint_path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    import torch  # type: ignore

    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {"model_state_dict": checkpoint}


def extract_checkpoint_state_dict(checkpoint_payload: dict[str, Any]) -> dict[str, Any]:
    state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint payload does not contain a valid state dict.")
    return state_dict


def remap_state_dict_to_model_state(
    source_state: dict[str, Any],
    target_model_state: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    filtered_state: dict[str, Any] = {}
    skipped_mismatches: list[str] = []
    for key, value in source_state.items():
        for target_key in _candidate_target_keys(key, target_model_state):
            target = target_model_state[target_key]
            if tuple(value.shape) != tuple(target.shape):
                skipped_mismatches.append(f"{key}->{target_key}")
                continue
            filtered_state[target_key] = value
    return filtered_state, skipped_mismatches


def interpolate_model_states(
    source_a: dict[str, Any],
    source_b: dict[str, Any],
    target_model_state: dict[str, Any],
    alpha: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    alpha = float(alpha)
    remapped_a, skipped_a = remap_state_dict_to_model_state(source_a, target_model_state)
    remapped_b, skipped_b = remap_state_dict_to_model_state(source_b, target_model_state)

    interpolated_state: dict[str, Any] = {}
    missing_in_b: list[str] = []
    shape_mismatches: list[str] = []
    for key, value_a in remapped_a.items():
        value_b = remapped_b.get(key)
        if value_b is None:
            missing_in_b.append(key)
            continue
        if tuple(value_a.shape) != tuple(value_b.shape):
            shape_mismatches.append(key)
            continue
        interpolated_state[key] = ((1.0 - alpha) * value_a) + (alpha * value_b)

    metadata = {
        "loaded_keys_a": len(remapped_a),
        "loaded_keys_b": len(remapped_b),
        "interpolated_keys": len(interpolated_state),
        "missing_in_b": missing_in_b,
        "shape_mismatch_keys": shape_mismatches,
        "skipped_shape_mismatch_keys_a": skipped_a,
        "skipped_shape_mismatch_keys_b": skipped_b,
    }
    return interpolated_state, metadata


def load_model_init_checkpoint(model, checkpoint_path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    checkpoint_file = Path(checkpoint_path)
    checkpoint = load_checkpoint_payload(checkpoint_file, map_location=map_location)
    state_dict = extract_checkpoint_state_dict(checkpoint)
    filtered_state, skipped_mismatches = remap_state_dict_to_model_state(state_dict, model.state_dict())

    load_result = model.load_state_dict(filtered_state, strict=False)
    return {
        "checkpoint_path": str(checkpoint_file),
        "loaded_keys": len(filtered_state),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "skipped_shape_mismatch_keys": skipped_mismatches,
    }


def load_external_backbone_only(model, checkpoint_path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    checkpoint_file = Path(checkpoint_path)
    checkpoint = load_checkpoint_payload(checkpoint_file, map_location=map_location)
    state_dict = extract_checkpoint_state_dict(checkpoint)

    # Filter state_dict to keep only backbone keys if they aren't already prefixed
    # or ensure they match backbone.* in target
    backbone_state: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            backbone_state[key] = value
        else:
            backbone_state[f"backbone.{key}"] = value

    # We only want to load into model.backbone
    target_state = model.state_dict()
    final_filtered: dict[str, Any] = {}
    skipped_mismatches: list[str] = []

    for key, value in backbone_state.items():
        if key in target_state:
            target = target_state[key]
            if tuple(value.shape) == tuple(target.shape):
                final_filtered[key] = value
            else:
                skipped_mismatches.append(f"{key} (shape mismatch)")

    load_result = model.load_state_dict(final_filtered, strict=False)
    return {
        "checkpoint_path": str(checkpoint_file),
        "mode": "external_backbone_only",
        "loaded_keys": len(final_filtered),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "skipped_shape_mismatch_keys": skipped_mismatches,
    }
