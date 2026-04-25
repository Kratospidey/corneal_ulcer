from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from checkpoint_utils import (
    extract_checkpoint_state_dict,
    interpolate_model_states,
    load_checkpoint_payload,
)
from config_utils import resolve_config, write_json, write_text
from data.dataset import build_dataloaders, build_datasets
from data.label_utils import get_task_definition
from data.split_utils import ensure_task_splits, load_manifest, load_split_dataframe
from data.transforms import build_transforms
from evaluation.calibration import compute_calibration
from evaluation.confusion import save_confusion_matrix
from evaluation.evaluate import run_inference
from evaluation.metrics import compute_classification_metrics
from evaluation.reports import save_metric_artifacts, write_experiment_report
from experiment_utils import prepare_output_dirs, resolve_device, setup_logging
from model_factory import create_model
from training.losses import build_loss, compute_class_weights


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Interpolate two checkpoints and evaluate them.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--official-checkpoint", required=True)
    parser.add_argument("--challenger-checkpoint", required=True)
    parser.add_argument("--official-label", default="official")
    parser.add_argument("--challenger-label", default="challenger")
    parser.add_argument("--experiment-tag", default="ckptinterp")
    parser.add_argument("--report-title", default="Checkpoint Interpolation")
    parser.add_argument(
        "--report-path",
        default="outputs/reports/model_improve/checkpoint_interpolation.md",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument(
        "--output-root",
        default="outputs/model_improve_2026-04-25/checkpoint_interpolation",
    )
    return parser


def _build_context(config_path: str, device: str):
    config = resolve_config(config_path)
    config["_config_path"] = config_path
    task_config = resolve_config(config["task_config"])
    split_config = resolve_config(config["split_config"])
    task_definition = get_task_definition(str(task_config["task_name"]))
    logger = setup_logging()
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
    split_path = Path(config.get("split_file", split_paths["holdout"]))
    manifest_df = load_manifest(split_config["manifest_path"])
    split_df = load_split_dataframe(split_path)
    transforms_by_split = build_transforms(
        int(config.get("image_size", 224)),
        train_profile=str(config.get("train_transform_profile", "default")),
    )
    datasets = build_datasets(
        manifest_df=manifest_df,
        split_df=split_df,
        label_column=task_definition.label_column,
        class_names=task_definition.class_names,
        transforms_by_split=transforms_by_split,
        preprocessing_mode=str(config.get("preprocessing_mode", "raw_rgb")),
        include_masks=bool(config.get("include_masks", False)),
    )
    loaders = build_dataloaders(
        datasets=datasets,
        batch_size=int(config.get("batch_size", 16)),
        num_workers=int(config.get("eval_num_workers", 0)),
        sampler=None,
        shuffle_train=False,
    )
    class_weights = None
    if bool(config.get("use_class_weights", True)):
        class_weights = compute_class_weights(task_definition.class_names, datasets["train"].class_counts()).to(device)
    criterion = build_loss(
        str(config.get("loss_name", "weighted_ce")),
        class_weights=class_weights,
        focal_gamma=float(config.get("focal_gamma", 2.0)),
        label_smoothing=float(config.get("label_smoothing", 0.0)),
        class_names=task_definition.class_names,
        label_counts=datasets["train"].class_counts(),
        logit_adjustment_tau=float(config.get("logit_adjustment_tau", 1.0)),
        class_balanced_beta=float(config.get("class_balanced_beta", 0.999)),
    )
    return {
        "config": config,
        "task_definition": task_definition,
        "datasets": datasets,
        "loaders": loaders,
        "criterion": criterion,
    }


def _evaluate_and_save(
    model,
    loaders,
    criterion,
    class_names,
    output_dirs: dict[str, Path],
    task_name: str,
    source_config_path: str,
    checkpoint_path: Path,
    split_name: str,
    device: str,
):
    payload = run_inference(
        model,
        loaders[split_name],
        device=device,
        criterion=criterion,
        progress_desc=f"Interp {split_name}",
        show_progress=False,
    )
    metrics_payload = compute_classification_metrics(
        payload["y_true"],
        payload["y_pred"],
        payload["probabilities"],
        class_names=class_names,
    )
    calibration_payload = compute_calibration(payload["probabilities"], payload["y_true"])
    save_metric_artifacts(
        evaluation_payload=payload,
        metrics_payload=metrics_payload,
        calibration_payload=calibration_payload,
        class_names=class_names,
        output_dirs=output_dirs,
        split_name=split_name,
        task_name=task_name,
        source_config_path=source_config_path,
        checkpoint_path=checkpoint_path,
    )
    save_confusion_matrix(
        payload["y_true"],
        payload["y_pred"],
        class_names,
        output_csv=output_dirs["confusion_matrices"] / f"{split_name}_confusion_matrix.csv",
        output_png=output_dirs["confusion_matrices"] / f"{split_name}_confusion_matrix.png",
    )
    merged = {**metrics_payload["metrics"], **calibration_payload}
    write_experiment_report(
        experiment_name=output_dirs["exported"].name,
        split_name=split_name,
        metrics=merged,
        output_path=output_dirs["reports"] / f"{split_name}_summary.md",
    )
    report = merged["classification_report"]
    return {
        "accuracy": float(merged["accuracy"]),
        "balanced_accuracy": float(merged["balanced_accuracy"]),
        "macro_f1": float(merged["macro_f1"]),
        "weighted_f1": float(merged["weighted_f1"]),
        "ece": float(merged["ece"]),
        "point_like_recall": float(report["point_like"]["recall"]),
        "point_flaky_mixed_recall": float(report["point_flaky_mixed"]["recall"]),
        "flaky_recall": float(report["flaky"]["recall"]),
    }


def _alpha_tag(alpha: float) -> str:
    return f"alpha{int(round(alpha * 100)):03d}"


def _write_summary(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    sorted_rows = sorted(rows, key=lambda row: (-row["val_balanced_accuracy"], -row["test_balanced_accuracy"], -row["test_macro_f1"]))
    lines = [
        f"# {title}",
        "",
        "| Alpha | Val BA | Test BA | Test Macro F1 | PL Recall | PFM Recall | Flaky Recall |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted_rows:
        lines.append(
            "| {alpha:.2f} | {val_balanced_accuracy:.4f} | {test_balanced_accuracy:.4f} | {test_macro_f1:.4f} | "
            "{test_point_like_recall:.4f} | {test_point_flaky_mixed_recall:.4f} | {test_flaky_recall:.4f} |".format(**row)
        )
    write_text(output_path, "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    import torch  # type: ignore

    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    context = _build_context(args.config, device=device)
    config = context["config"]
    task_definition = context["task_definition"]
    output_root = Path(args.output_root)
    official_payload = load_checkpoint_payload(args.official_checkpoint, map_location="cpu")
    challenger_payload = load_checkpoint_payload(args.challenger_checkpoint, map_location="cpu")
    official_state = extract_checkpoint_state_dict(official_payload)
    challenger_state = extract_checkpoint_state_dict(challenger_payload)
    results: list[dict[str, Any]] = []

    for alpha in [float(value) for value in args.alphas]:
        experiment_name = (
            "pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__"
            f"{args.experiment_tag}_{_alpha_tag(alpha)}__holdout_v1__seed42"
        )
        output_dirs = prepare_output_dirs(experiment_name, output_root=output_root)
        model = create_model(config["model"], num_classes=len(task_definition.class_names)).to(device)
        interpolated_state, interpolation_metadata = interpolate_model_states(
            official_state,
            challenger_state,
            model.state_dict(),
            alpha=alpha,
        )
        load_result = model.load_state_dict(interpolated_state, strict=False)
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "source_checkpoints": {
                "official": str(args.official_checkpoint),
                "challenger": str(args.challenger_checkpoint),
            },
            "alpha": alpha,
            "interpolation_metadata": interpolation_metadata,
            "missing_keys_after_load": list(load_result.missing_keys),
            "unexpected_keys_after_load": list(load_result.unexpected_keys),
        }
        checkpoint_metadata = {
            "source_checkpoints": checkpoint_payload["source_checkpoints"],
            "alpha": alpha,
            "interpolation_metadata": interpolation_metadata,
            "missing_keys_after_load": list(load_result.missing_keys),
            "unexpected_keys_after_load": list(load_result.unexpected_keys),
        }
        checkpoint_path = output_dirs["exported"] / "best.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload, checkpoint_path)
        write_json(output_dirs["reports"] / "interpolation_metadata.json", checkpoint_metadata)

        val_metrics = _evaluate_and_save(
            model=model,
            loaders=context["loaders"],
            criterion=context["criterion"],
            class_names=task_definition.class_names,
            output_dirs=output_dirs,
            task_name=task_definition.task_name,
            source_config_path=args.config,
            checkpoint_path=checkpoint_path,
            split_name="val",
            device=device,
        )
        test_metrics = _evaluate_and_save(
            model=model,
            loaders=context["loaders"],
            criterion=context["criterion"],
            class_names=task_definition.class_names,
            output_dirs=output_dirs,
            task_name=task_definition.task_name,
            source_config_path=args.config,
            checkpoint_path=checkpoint_path,
            split_name="test",
            device=device,
        )
        results.append(
            {
                "alpha": alpha,
                "experiment_name": experiment_name,
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "test_ece": test_metrics["ece"],
                "test_point_like_recall": test_metrics["point_like_recall"],
                "test_point_flaky_mixed_recall": test_metrics["point_flaky_mixed_recall"],
                "test_flaky_recall": test_metrics["flaky_recall"],
                "checkpoint_path": str(checkpoint_path),
            }
        )

    write_json(output_root / "interpolation_results.json", results)
    _write_summary(results, Path(args.report_path), args.report_title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
