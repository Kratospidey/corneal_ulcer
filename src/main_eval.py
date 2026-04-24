from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from console_utils import (
    emit_artifact_summary,
    emit_dataset_summary,
    emit_figure_summary,
    emit_model_summary,
    emit_run_header,
    emit_split_metrics,
    get_console,
)
from config_utils import resolve_config
from data.dataset import build_dataloaders, build_datasets
from data.label_utils import get_task_definition
from data.split_utils import ensure_task_splits, load_manifest, load_split_dataframe
from data.transforms import build_transforms
from evaluation.calibration import compute_calibration
from evaluation.confusion import save_confusion_matrix
from evaluation.evaluate import run_inference
from evaluation.metrics import compute_classification_metrics
from evaluation.paper_figures import generate_paper_figure_bundle
from evaluation.reports import save_metric_artifacts, write_experiment_report
from experiment_utils import build_experiment_name, prepare_output_dirs, resolve_device, setup_logging
from model_factory import create_model
from training.losses import build_loss, compute_class_weights


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Stage 3 baseline evaluation entrypoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging()
    console = get_console()
    config = resolve_config(args.config)
    task_config = resolve_config(config["task_config"])
    split_config = resolve_config(config["split_config"])
    task_definition = get_task_definition(str(task_config["task_name"]))
    experiment_name = build_experiment_name({**config, "task_name": task_definition.task_name})
    output_dirs = prepare_output_dirs(experiment_name, output_root=config.get("output_root", "outputs"))
    device = resolve_device(args.device)
    emit_run_header(
        console,
        title="Pattern Evaluation Run",
        experiment_name=experiment_name,
        task_name=task_definition.task_name,
        device=device,
        config_path=args.config,
        output_root=config.get("output_root", "outputs"),
    )
    torch_num_threads = int(config.get("torch_num_threads", 0) or 0)
    if torch_num_threads > 0:
        import torch  # type: ignore

        torch.set_num_threads(torch_num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, min(torch_num_threads, 4)))
            except RuntimeError:
                pass

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
    manifest_df = load_manifest(split_config["manifest_path"])
    split_df = load_split_dataframe(config.get("split_file", split_paths["holdout"]))
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
        num_workers=int(config.get("num_workers", 4)),
        sampler=None,
        shuffle_train=False,
    )
    emit_dataset_summary(
        console,
        datasets=datasets,
        class_names=task_definition.class_names,
        batch_size=int(config.get("batch_size", 16)),
        epochs=int(config.get("epochs", 10)),
        preprocessing_mode=str(config.get("preprocessing_mode", "raw_rgb")),
        train_transform_profile=str(config.get("train_transform_profile", "default")),
        sampler_name="none",
    )
    model = create_model(config["model"], num_classes=len(task_definition.class_names)).to(device)

    import torch  # type: ignore

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    emit_model_summary(console, model=model, training_config=config, class_weights=class_weights)

    evaluation_payload = run_inference(
        model,
        loaders[args.split],
        device=device,
        criterion=criterion,
        progress_desc=f"Evaluating {args.split}",
        show_progress=bool(config.get("show_progress", True)),
    )
    metrics_payload = compute_classification_metrics(
        evaluation_payload["y_true"],
        evaluation_payload["y_pred"],
        evaluation_payload["probabilities"],
        class_names=task_definition.class_names,
    )
    calibration_payload = compute_calibration(evaluation_payload["probabilities"], evaluation_payload["y_true"])
    save_metric_artifacts(
        evaluation_payload,
        metrics_payload,
        calibration_payload,
        task_definition.class_names,
        output_dirs,
        args.split,
        task_name=task_definition.task_name,
        source_config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    save_confusion_matrix(
        evaluation_payload["y_true"],
        evaluation_payload["y_pred"],
        task_definition.class_names,
        output_csv=output_dirs["confusion_matrices"] / f"{args.split}_confusion_matrix.csv",
        output_png=output_dirs["confusion_matrices"] / f"{args.split}_confusion_matrix.png",
    )
    write_experiment_report(
        experiment_name=experiment_name,
        split_name=args.split,
        metrics={**metrics_payload["metrics"], **calibration_payload},
        output_path=output_dirs["reports"] / f"{args.split}_summary.md",
    )
    merged_metrics = {**metrics_payload["metrics"], **calibration_payload}
    emit_split_metrics(
        console,
        split_name=args.split,
        metrics=merged_metrics,
        class_names=task_definition.class_names,
        report_path=output_dirs["reports"] / f"{args.split}_summary.md",
    )
    emit_artifact_summary(
        console,
        checkpoint_path=args.checkpoint,
        exported_checkpoint_path=None,
        metrics_dir=output_dirs["metrics"],
        reports_dir=output_dirs["reports"],
    )
    if args.split == "test" and bool(config.get("auto_generate_paper_figures", True)):
        try:
            figure_bundle = generate_paper_figure_bundle(
                train_config=config,
                checkpoint_path=args.checkpoint,
                output_root=Path(config.get("output_root", "outputs")) / "paper_figures",
                device=str(config.get("paper_figures_device", device)),
                xai_count=int(config.get("paper_figures_xai_count", 6)),
                logger=logger,
            )
            emit_figure_summary(
                console,
                figure_output_dir=figure_bundle["output_dir"],
                figure_manifest_path=figure_bundle["manifest_path"],
            )
        except Exception as exc:
            logger.warning("Automatic paper figure generation failed: %s", exc)
    logger.info("Saved evaluation artifacts for %s split=%s", experiment_name, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
