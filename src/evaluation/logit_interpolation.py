from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

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
    parser = ArgumentParser(description="Blend official and challenger logits.")
    parser.add_argument("--official-config", required=True)
    parser.add_argument("--official-checkpoint", required=True)
    parser.add_argument("--challenger-config", required=True)
    parser.add_argument("--challenger-checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument(
        "--output-root",
        default="outputs/model_improve_2026-04-25/logit_interpolation_official_vs_v2w005",
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
    split_df = load_split_dataframe(Path(config.get("split_file", split_paths["holdout"])))
    manifest_df = load_manifest(split_config["manifest_path"])
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
    return {"config": config, "task_definition": task_definition, "loaders": loaders, "criterion": criterion}


def _run_checkpoint(config_path: str, checkpoint_path: str, device: str, split_name: str):
    import torch  # type: ignore

    context = _build_context(config_path, device=device)
    model = create_model(context["config"]["model"], num_classes=len(context["task_definition"].class_names)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    payload = run_inference(
        model,
        context["loaders"][split_name],
        device=device,
        criterion=context["criterion"],
        progress_desc=f"{Path(checkpoint_path).stem}:{split_name}",
        show_progress=False,
    )
    return payload, context


def _blend_payloads(official_payload: dict[str, Any], challenger_payload: dict[str, Any], alpha: float):
    import numpy as np  # type: ignore

    official_rows = {row["image_id"]: row for row in official_payload["prediction_rows"]}
    challenger_rows = {row["image_id"]: row for row in challenger_payload["prediction_rows"]}
    shared_ids = list(official_rows.keys())
    if shared_ids != list(challenger_rows.keys()):
        raise ValueError("Official and challenger prediction order does not match.")

    official_logits = np.asarray(official_payload["logits"])
    challenger_logits = np.asarray(challenger_payload["logits"])
    blended_logits = ((1.0 - alpha) * official_logits) + (alpha * challenger_logits)
    exp_logits = np.exp(blended_logits - blended_logits.max(axis=1, keepdims=True))
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    preds = probabilities.argmax(axis=1).tolist()

    prediction_rows = []
    for row_index, image_id in enumerate(shared_ids):
        base_row = dict(official_rows[image_id])
        base_row["predicted_index"] = int(preds[row_index])
        base_row["confidence"] = float(probabilities[row_index].max())
        base_row["logits"] = [float(value) for value in blended_logits[row_index].tolist()]
        prediction_rows.append(base_row)
    return {
        "loss": None,
        "y_true": list(official_payload["y_true"]),
        "y_pred": preds,
        "probabilities": probabilities,
        "logits": blended_logits,
        "prediction_rows": prediction_rows,
    }


def _evaluate_payload(payload: dict[str, Any], class_names):
    metrics_payload = compute_classification_metrics(
        payload["y_true"],
        payload["y_pred"],
        payload["probabilities"],
        class_names=class_names,
    )
    calibration_payload = compute_calibration(payload["probabilities"], payload["y_true"])
    merged = {**metrics_payload["metrics"], **calibration_payload}
    report = merged["classification_report"]
    return {
        "metrics_payload": metrics_payload,
        "calibration_payload": calibration_payload,
        "summary": {
            "accuracy": float(merged["accuracy"]),
            "balanced_accuracy": float(merged["balanced_accuracy"]),
            "macro_f1": float(merged["macro_f1"]),
            "weighted_f1": float(merged["weighted_f1"]),
            "ece": float(merged["ece"]),
            "point_like_recall": float(report["point_like"]["recall"]),
            "point_flaky_mixed_recall": float(report["point_flaky_mixed"]["recall"]),
            "flaky_recall": float(report["flaky"]["recall"]),
        },
    }


def _write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    sorted_rows = sorted(rows, key=lambda row: (-row["val_balanced_accuracy"], -row["test_balanced_accuracy"], -row["test_macro_f1"]))
    lines = [
        "# Logit Interpolation: Official vs v2w005",
        "",
        "Deployment-only results.",
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
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    output_root = Path(args.output_root)
    context = _build_context(args.official_config, device=device)
    class_names = context["task_definition"].class_names
    task_name = context["task_definition"].task_name
    results: list[dict[str, Any]] = []

    source_payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for source_name, config_path, checkpoint_path in (
        ("official", args.official_config, args.official_checkpoint),
        ("challenger", args.challenger_config, args.challenger_checkpoint),
    ):
        source_payloads[source_name] = {}
        for split_name in ("val", "test"):
            payload, _ = _run_checkpoint(config_path, checkpoint_path, device=device, split_name=split_name)
            source_payloads[source_name][split_name] = payload

    for alpha in [float(value) for value in args.alphas]:
        experiment_name = (
            "pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__"
            f"logitinterp_v2w005_alpha{int(round(alpha * 100)):03d}__holdout_v1__seed42"
        )
        output_dirs = prepare_output_dirs(experiment_name, output_root=output_root)
        split_summaries: dict[str, dict[str, float]] = {}
        for split_name in ("val", "test"):
            blended_payload = _blend_payloads(
                source_payloads["official"][split_name],
                source_payloads["challenger"][split_name],
                alpha=alpha,
            )
            evaluated = _evaluate_payload(blended_payload, class_names=class_names)
            save_metric_artifacts(
                evaluation_payload=blended_payload,
                metrics_payload=evaluated["metrics_payload"],
                calibration_payload=evaluated["calibration_payload"],
                class_names=class_names,
                output_dirs=output_dirs,
                split_name=split_name,
                task_name=task_name,
                source_config_path=args.official_config,
                checkpoint_path=f"logit_interpolation:official={args.official_checkpoint};challenger={args.challenger_checkpoint};alpha={alpha:.2f}",
            )
            save_confusion_matrix(
                blended_payload["y_true"],
                blended_payload["y_pred"],
                class_names,
                output_csv=output_dirs["confusion_matrices"] / f"{split_name}_confusion_matrix.csv",
                output_png=output_dirs["confusion_matrices"] / f"{split_name}_confusion_matrix.png",
            )
            merged = {**evaluated["metrics_payload"]["metrics"], **evaluated["calibration_payload"]}
            write_experiment_report(
                experiment_name=experiment_name,
                split_name=split_name,
                metrics=merged,
                output_path=output_dirs["reports"] / f"{split_name}_summary.md",
            )
            split_summaries[split_name] = evaluated["summary"]

        results.append(
            {
                "alpha": alpha,
                "experiment_name": experiment_name,
                "val_balanced_accuracy": split_summaries["val"]["balanced_accuracy"],
                "val_macro_f1": split_summaries["val"]["macro_f1"],
                "test_balanced_accuracy": split_summaries["test"]["balanced_accuracy"],
                "test_macro_f1": split_summaries["test"]["macro_f1"],
                "test_accuracy": split_summaries["test"]["accuracy"],
                "test_weighted_f1": split_summaries["test"]["weighted_f1"],
                "test_ece": split_summaries["test"]["ece"],
                "test_point_like_recall": split_summaries["test"]["point_like_recall"],
                "test_point_flaky_mixed_recall": split_summaries["test"]["point_flaky_mixed_recall"],
                "test_flaky_recall": split_summaries["test"]["flaky_recall"],
            }
        )

    write_json(output_root / "logit_interpolation_results.json", results)
    _write_summary(results, Path("outputs/reports/model_improve/logit_interpolation_official_vs_v2w005.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
