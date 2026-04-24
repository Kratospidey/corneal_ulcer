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
from checkpoint_utils import load_model_init_checkpoint
from config_utils import resolve_config, write_json
from data.dataset import build_dataloaders, build_datasets
from data.label_utils import get_task_definition
from data.split_utils import ensure_task_splits, load_manifest, load_split_dataframe
from data.transforms import build_transforms
from experiment_utils import build_experiment_name, prepare_output_dirs, resolve_device, set_seed, setup_logging
from model_factory import create_model
from provenance_utils import build_data_provenance
from training.losses import build_loss, compute_class_weights
from training.optim_utils import build_optimizer, build_scheduler
from training.samplers import build_sampler
from training.train import run_training_pipeline


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Stage 3 baseline training entrypoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging()
    console = get_console()
    config = resolve_config(args.config)
    config["_config_path"] = args.config
    task_config = resolve_config(config["task_config"])
    split_config = resolve_config(config["split_config"])
    task_name = str(task_config["task_name"])
    task_definition = get_task_definition(task_name)

    set_seed(int(config.get("seed", 42)))
    torch_num_threads = int(config.get("torch_num_threads", 0) or 0)
    if torch_num_threads > 0:
        import torch  # type: ignore

        torch.set_num_threads(torch_num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, min(torch_num_threads, 4)))
            except RuntimeError:
                pass
    experiment_name = build_experiment_name({**config, "task_name": task_name})
    output_dirs = prepare_output_dirs(experiment_name, output_root=config.get("output_root", "outputs"))
    device = resolve_device(args.device)
    logger.info("Training %s on %s using device=%s", experiment_name, task_name, device)
    emit_run_header(
        console,
        title="Pattern Training Run",
        experiment_name=experiment_name,
        task_name=task_name,
        device=device,
        config_path=args.config,
        output_root=config.get("output_root", "outputs"),
    )

    split_paths = ensure_task_splits(
        manifest_path=split_config["manifest_path"],
        duplicate_csv_path=split_config["duplicate_candidates_path"],
        split_dir=split_config["split_dir"],
        task_name=task_name,
        label_column=task_definition.label_column,
        holdout_seed=int(split_config.get("holdout", {}).get("seed", 42)),
        cv_seed=int(split_config.get("repeated_cv", {}).get("seed", 42)),
        logger=logger,
    )
    split_path = Path(config.get("split_file", split_paths["holdout"]))
    data_provenance = build_data_provenance(
        manifest_path=split_config["manifest_path"],
        split_file=split_path,
    )

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
    sampler = build_sampler(
        datasets["train"],
        str(config.get("sampler", "none")),
        sampler_temperature=float(config.get("sampler_temperature", 1.0)),
    )
    loaders = build_dataloaders(
        datasets=datasets,
        batch_size=int(config.get("batch_size", 16)),
        num_workers=int(config.get("num_workers", 4)),
        sampler=sampler,
    )
    emit_dataset_summary(
        console,
        datasets=datasets,
        class_names=task_definition.class_names,
        batch_size=int(config.get("batch_size", 16)),
        epochs=int(config.get("epochs", 10)),
        preprocessing_mode=str(config.get("preprocessing_mode", "raw_rgb")),
        train_transform_profile=str(config.get("train_transform_profile", "default")),
        sampler_name=str(config.get("sampler", "none")),
    )

    model = create_model(config["model"], num_classes=len(task_definition.class_names)).to(device)
    init_checkpoint_summary = None
    init_checkpoint_path = config.get("init_checkpoint_path")
    if init_checkpoint_path:
        init_checkpoint_summary = load_model_init_checkpoint(model, init_checkpoint_path, map_location=device)
        logger.info(
            "Initialized %s from %s with %d loaded keys (%d missing, %d mismatched).",
            experiment_name,
            init_checkpoint_summary["checkpoint_path"],
            init_checkpoint_summary["loaded_keys"],
            len(init_checkpoint_summary["missing_keys"]),
            len(init_checkpoint_summary["skipped_shape_mismatch_keys"]),
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
    emit_model_summary(console, model=model, training_config=config, class_weights=class_weights)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    results = run_training_pipeline(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        class_names=task_definition.class_names,
        device=device,
        training_config=config,
        output_dirs=output_dirs,
        experiment_name=experiment_name,
        console=console,
    )
    write_json(
        output_dirs["reports"] / "run_metadata.json",
        {
            "config_path": args.config,
            "device": device,
            "data_provenance": data_provenance,
            "init_checkpoint": init_checkpoint_summary,
            **results["splits"],
        },
    )
    for split_name, metrics in results["splits"].items():
        emit_split_metrics(
            console,
            split_name=split_name,
            metrics=metrics,
            class_names=task_definition.class_names,
            report_path=output_dirs["reports"] / f"{split_name}_summary.md",
        )
    emit_artifact_summary(
        console,
        checkpoint_path=results["checkpoint_path"],
        exported_checkpoint_path=results.get("exported_checkpoint_path"),
        metrics_dir=output_dirs["metrics"],
        reports_dir=output_dirs["reports"],
    )
    figure_bundle = results.get("figure_bundle")
    if isinstance(figure_bundle, dict) and figure_bundle.get("output_dir"):
        emit_figure_summary(
            console,
            figure_output_dir=figure_bundle["output_dir"],
            figure_manifest_path=figure_bundle.get("manifest_path"),
        )
    elif isinstance(figure_bundle, dict) and figure_bundle.get("error"):
        logger.warning("Automatic paper figure generation failed: %s", figure_bundle["error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
