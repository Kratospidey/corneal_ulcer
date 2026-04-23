from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

from config_utils import resolve_config, write_json
from data.dataset import build_dataloaders, build_datasets
from data.label_utils import get_task_definition
from data.split_utils import ensure_task_splits, load_manifest, load_split_dataframe
from data.transforms import build_transforms
from evaluation.confusion import save_confusion_matrix
from evaluation.evaluate import run_inference
from evaluation.metrics import compute_classification_metrics
from evaluation.prediction_contract import (
    build_prediction_provenance,
    build_prediction_row,
    probability_column_names,
    validate_prediction_provenance,
    validate_prediction_rows,
)
from evaluation.promotion import build_promotion_reference_block
from evaluation.reports import write_experiment_report
from experiment_utils import prepare_output_dirs, resolve_device, setup_logging, write_csv_rows
from model_factory import create_model
from training.losses import build_loss, compute_class_weights


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Config-driven late fusion evaluation for pattern models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser


def _build_component_prediction_tables(component: dict[str, Any], split_name: str, device: str, logger):
    import torch  # type: ignore

    train_config = resolve_config(component["config_path"])
    task_config = resolve_config(train_config["task_config"])
    split_config = resolve_config(train_config["split_config"])
    task_definition = get_task_definition(str(task_config["task_name"]))

    split_paths = ensure_task_splits(
        manifest_path=split_config["manifest_path"],
        duplicate_csv_path=split_config["duplicate_candidates_path"],
        split_dir=split_config["split_dir"],
        task_name=task_definition.task_name,
        label_column=task_definition.label_column,
        holdout_seed=int(split_config.get("holdout", {}).get("seed", 42)),
        cv_seed=int(split_config.get("repeated_cv", {}).get("seed", 42)),
        logger=logger,
    )
    split_df = load_split_dataframe(train_config.get("split_file", split_paths["holdout"]))
    manifest_df = load_manifest(split_config["manifest_path"])
    transforms_by_split = build_transforms(
        int(train_config.get("image_size", 224)),
        train_profile=str(train_config.get("train_transform_profile", "default")),
    )
    datasets = build_datasets(
        manifest_df=manifest_df,
        split_df=split_df,
        label_column=task_definition.label_column,
        class_names=task_definition.class_names,
        transforms_by_split=transforms_by_split,
        preprocessing_mode=str(component.get("preprocessing_mode", train_config.get("preprocessing_mode", "raw_rgb"))),
        include_masks=bool(train_config.get("include_masks", False)),
    )
    loaders = build_dataloaders(
        datasets=datasets,
        batch_size=int(train_config.get("batch_size", 16)),
        num_workers=int(train_config.get("num_workers", 4)),
        sampler=None,
    )

    model = create_model(train_config["model"], num_classes=len(task_definition.class_names)).to(device)
    checkpoint = torch.load(component["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    class_weights = None
    if bool(train_config.get("use_class_weights", True)):
        class_weights = compute_class_weights(task_definition.class_names, datasets["train"].class_counts()).to(device)
    criterion = build_loss(
        str(train_config.get("loss_name", "weighted_ce")),
        class_weights=class_weights,
        focal_gamma=float(train_config.get("focal_gamma", 2.0)),
        label_smoothing=float(train_config.get("label_smoothing", 0.0)),
    )
    payload = run_inference(model, loaders[split_name], device=device, criterion=criterion)

    prediction_rows: list[dict[str, Any]] = []
    probabilities = payload["probabilities"]
    for row_index, base_row in enumerate(payload["prediction_rows"]):
        probs = [float(value) for value in probabilities[row_index].tolist()]
        prediction_rows.append(
            build_prediction_row(
                base_row=base_row,
                class_names=task_definition.class_names,
                probabilities=probs,
                extras={
                    "raw_image_path": str(base_row.get("raw_image_path", "")),
                    "cornea_mask_path": str(base_row.get("cornea_mask_path", "")),
                    "ulcer_mask_path": str(base_row.get("ulcer_mask_path", "")),
                    "confidence": float(base_row.get("confidence", max(probs))),
                    "true_label": task_definition.class_names[int(payload["y_true"][row_index])],
                    "pred_label": task_definition.class_names[int(payload["y_pred"][row_index])],
                    "correct": bool(payload["y_true"][row_index] == payload["y_pred"][row_index]),
                },
            )
        )
    prediction_provenance = build_prediction_provenance(
        task_name=task_definition.task_name,
        class_names=task_definition.class_names,
        split_name=split_name,
        source_config_path=component["config_path"],
        checkpoint_path=component["checkpoint_path"],
    )
    validate_prediction_rows(prediction_rows, class_names=task_definition.class_names, split_name=split_name)
    validate_prediction_provenance(
        prediction_provenance,
        task_name=task_definition.task_name,
        class_names=task_definition.class_names,
        split_name=split_name,
    )
    prediction_lookup = {str(row["image_id"]): row for row in prediction_rows}
    return {
        "lookup": prediction_lookup,
        "rows": prediction_rows,
        "provenance": prediction_provenance,
        "class_names": task_definition.class_names,
        "split_file": str(train_config.get("split_file", split_paths["holdout"])),
    }


def _validate_component_tables(
    component_payloads: list[dict[str, Any]],
    task_name: str,
    class_names: list[str] | tuple[str, ...],
    split_name: str,
) -> list[str]:
    if not component_payloads:
        raise ValueError("Late fusion requires at least one component payload.")

    expected_probability_columns = probability_column_names(class_names)
    expected_row_schema = list(component_payloads[0]["rows"][0].keys())
    expected_image_ids = set(component_payloads[0]["lookup"].keys())

    for payload in component_payloads:
        validate_prediction_rows(payload["rows"], class_names=class_names, split_name=split_name)
        validate_prediction_provenance(
            payload["provenance"],
            task_name=task_name,
            class_names=class_names,
            split_name=split_name,
        )

        row_schema = list(payload["rows"][0].keys())
        if row_schema != expected_row_schema:
            raise ValueError(f"Fusion component schema mismatch: expected {expected_row_schema}, got {row_schema}")
        if payload["provenance"].get("probability_columns") != expected_probability_columns:
            raise ValueError(
                "Fusion component probability-column provenance mismatch: "
                f"expected {expected_probability_columns}, got {payload['provenance'].get('probability_columns')}"
            )

        image_ids = set(payload["lookup"].keys())
        if image_ids != expected_image_ids:
            missing_ids = sorted(expected_image_ids.difference(image_ids))
            extra_ids = sorted(image_ids.difference(expected_image_ids))
            raise ValueError(
                "Fusion component sample coverage mismatch: "
                f"missing={missing_ids[:5]} extra={extra_ids[:5]}"
            )

    ordered_ids = sorted(expected_image_ids)
    reference_lookup = component_payloads[0]["lookup"]
    for payload in component_payloads[1:]:
        for image_id in ordered_ids:
            if int(payload["lookup"][image_id]["target_index"]) != int(reference_lookup[image_id]["target_index"]):
                raise ValueError(f"Fusion component target_index mismatch for image_id '{image_id}'.")
    return ordered_ids


def _generate_weight_candidates(num_models: int, step: float) -> list[list[float]]:
    if num_models <= 1:
        return [[1.0]]
    scaled = int(round(1.0 / step))
    results: list[list[float]] = []

    def _walk(prefix: list[int], remaining_slots: int, remaining_units: int) -> None:
        if remaining_slots == 1:
            results.append([*(value / scaled for value in prefix), remaining_units / scaled])
            return
        for units in range(remaining_units + 1):
            _walk([*prefix, units], remaining_slots - 1, remaining_units - units)

    _walk([], num_models, scaled)
    return [candidate for candidate in results if abs(sum(candidate) - 1.0) < 1e-6 and any(value > 0 for value in candidate)]


def _weighted_fuse(probabilities_by_model: list[list[float]], weights: list[float]) -> list[float]:
    normalizer = sum(weights) or 1.0
    fused = [0.0 for _ in probabilities_by_model[0]]
    for model_probs, weight in zip(probabilities_by_model, weights, strict=False):
        for class_index, value in enumerate(model_probs):
            fused[class_index] += float(weight) * float(value)
    return [value / normalizer for value in fused]


def _fuse_tables(
    component_tables: list[dict[str, dict[str, Any]]],
    ordered_ids: list[str],
    class_names: list[str] | tuple[str, ...],
    weights: list[float],
) -> tuple[list[dict[str, Any]], list[int], list[int], list[list[float]]]:
    fused_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []

    for image_id in ordered_ids:
        entries = [table[image_id] for table in component_tables]
        probs = _weighted_fuse(
            [[float(entry[column]) for column in probability_column_names(class_names)] for entry in entries],
            weights,
        )
        pred_index = max(range(len(probs)), key=lambda idx: probs[idx])
        target_index = int(entries[0]["target_index"])
        y_true.append(target_index)
        y_pred.append(pred_index)
        probabilities.append(probs)
        fused_rows.append(
            build_prediction_row(
                base_row={
                    "image_id": image_id,
                    "split": entries[0]["split"],
                    "target_index": target_index,
                    "predicted_index": pred_index,
                },
                class_names=class_names,
                probabilities=probs,
                extras={
                    "confidence": max(probs),
                    "raw_image_path": entries[0].get("raw_image_path", ""),
                    "cornea_mask_path": entries[0].get("cornea_mask_path", ""),
                    "ulcer_mask_path": entries[0].get("ulcer_mask_path", ""),
                    "true_label": class_names[target_index],
                    "pred_label": class_names[pred_index],
                    "correct": bool(target_index == pred_index),
                },
            )
        )
    return fused_rows, y_true, y_pred, probabilities


def _select_weights(
    component_tables: list[dict[str, dict[str, Any]]],
    ordered_ids: list[str],
    class_names: list[str] | tuple[str, ...],
    weight_search_config: dict[str, Any],
) -> list[float]:
    fixed = weight_search_config.get("fixed_weights")
    if fixed:
        return [float(value) for value in fixed]

    step = float(weight_search_config.get("grid_step", 0.05))
    best_weights = [1.0 / len(component_tables) for _ in component_tables]
    best_score = float("-inf")
    for candidate in _generate_weight_candidates(len(component_tables), step):
        _, y_true, y_pred, probabilities = _fuse_tables(component_tables, ordered_ids, class_names, candidate)
        metrics = compute_classification_metrics(y_true, y_pred, probabilities, class_names)["metrics"]
        score = float(metrics["balanced_accuracy"])
        if score > best_score:
            best_score = score
            best_weights = candidate
    return best_weights


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging()
    config = resolve_config(args.config)
    device = resolve_device(args.device)
    task_definition = get_task_definition(str(resolve_config(config["task_config"])["task_name"]))
    output_dirs = prepare_output_dirs(str(config["experiment_name"]), output_root=config.get("output_root", "outputs"))

    component_payloads_by_split: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    split_file = None
    for component in config["inference"]["models"]:
        for split_name in ("val", "test"):
            payload = _build_component_prediction_tables(component, split_name=split_name, device=device, logger=logger)
            class_names = payload["class_names"]
            split_file = payload["split_file"]
            component_payloads_by_split[split_name].append(payload)

    if tuple(class_names) != tuple(task_definition.class_names):
        raise ValueError("Fusion components do not match the requested task class order.")

    ordered_ids_by_split = {
        split_name: _validate_component_tables(
            component_payloads=component_payloads_by_split[split_name],
            task_name=task_definition.task_name,
            class_names=class_names,
            split_name=split_name,
        )
        for split_name in ("val", "test")
    }
    component_tables_by_split = {
        split_name: [payload["lookup"] for payload in component_payloads_by_split[split_name]]
        for split_name in ("val", "test")
    }

    weights = _select_weights(
        component_tables_by_split["val"],
        ordered_ids_by_split["val"],
        class_names=class_names,
        weight_search_config=dict(config["inference"].get("weight_search", {})),
    )

    summary_rows: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        fused_rows, y_true, y_pred, probabilities = _fuse_tables(
            component_tables_by_split[split_name],
            ordered_ids_by_split[split_name],
            class_names,
            weights,
        )
        prediction_provenance = build_prediction_provenance(
            task_name=task_definition.task_name,
            class_names=class_names,
            split_name=split_name,
            source_config_path=args.config,
        )
        prediction_provenance["component_names"] = [component["name"] for component in config["inference"]["models"]]
        prediction_provenance["probability_fusion_weights"] = {
            component["name"]: float(weight)
            for component, weight in zip(config["inference"]["models"], weights, strict=False)
        }
        validate_prediction_rows(fused_rows, class_names=class_names, split_name=split_name)
        validate_prediction_provenance(
            prediction_provenance,
            task_name=task_definition.task_name,
            class_names=class_names,
            split_name=split_name,
        )
        metrics = compute_classification_metrics(y_true, y_pred, probabilities, class_names)["metrics"]
        metrics.update(
            build_promotion_reference_block(
                metrics,
                task_name=task_definition.task_name,
                promotion_config=config.get("promotion_reference_config"),
            )
        )
        metrics.update(
            {
                "artifact_path": str(output_dirs["reports"] / f"{split_name}_summary.md"),
                "split_file": split_file,
                "seed": config.get("seed"),
                "view_weights": {
                    component["name"]: float(weight)
                    for component, weight in zip(config["inference"]["models"], weights, strict=False)
                },
                "normalization_mode": config["inference"].get("normalization_mode", "probability"),
                "decision_rule": config["inference"].get("decision_rule", "weighted_average"),
            }
        )
        write_json(output_dirs["metrics"] / f"{split_name}_metrics.json", metrics)
        write_csv_rows(output_dirs["predictions"] / f"{split_name}_predictions.csv", fused_rows)
        write_json(output_dirs["predictions"] / f"{split_name}_prediction_metadata.json", prediction_provenance)
        save_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            output_csv=output_dirs["confusion_matrices"] / f"{split_name}_confusion_matrix.csv",
            output_png=output_dirs["confusion_matrices"] / f"{split_name}_confusion_matrix.png",
        )
        write_experiment_report(
            experiment_name=str(config["experiment_name"]),
            split_name=split_name,
            metrics=metrics,
            output_path=output_dirs["reports"] / f"{split_name}_summary.md",
            report_context={
                "artifact_path": output_dirs["reports"] / f"{split_name}_summary.md",
                "split_file": split_file,
                "seed": config.get("seed"),
            },
        )
        summary_rows.append(
            {
                "split": split_name,
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "delta_vs_official_single_ba": metrics.get("delta_vs_official_single_ba"),
                "delta_vs_deployed_rule_ba": metrics.get("delta_vs_deployed_rule_ba"),
            }
        )

    write_json(
        output_dirs["reports"] / "late_fusion_metadata.json",
        {
            "config_path": args.config,
            "device": device,
            "weights": {
                component["name"]: float(weight)
                for component, weight in zip(config["inference"]["models"], weights, strict=False)
            },
            "split_file": split_file,
            "summary": summary_rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
