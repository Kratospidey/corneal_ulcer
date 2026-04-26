from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil

from config_utils import resolve_config, write_json
from evaluation.calibration import compute_calibration
from evaluation.confusion import save_confusion_matrix
from evaluation.evaluate import run_inference
from evaluation.metrics import compute_classification_metrics
from evaluation.paper_figures import generate_paper_figure_bundle
from evaluation.reports import (
    save_metric_artifacts,
    write_experiment_report,
)
from experiment_utils import write_csv_rows
from training.trainer import fit_model


def run_training_pipeline(
    model,
    loaders,
    criterion,
    optimizer,
    scheduler,
    class_names: list[str] | tuple[str, ...],
    device: str,
    training_config: dict[str, Any],
    output_dirs: dict[str, Path],
    experiment_name: str,
    console=None,
    teacher_model=None,
    distillation_config: dict[str, Any] | None = None,
):
    checkpoint_path = output_dirs["models"] / "best.pt"
    show_progress = bool(training_config.get("show_progress", True))
    training_result = fit_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        class_names=class_names,
        device=device,
        training_config=training_config,
        checkpoint_path=checkpoint_path,
        console=console,
        teacher_model=teacher_model,
        distillation_config=distillation_config,
    )
    write_csv_rows(output_dirs["metrics"] / "history.csv", training_result.history)
    write_json(
        output_dirs["metrics"] / "training_summary.json",
        {
            "best_metric": training_result.best_metric,
            "best_epoch": training_result.best_epoch,
            "checkpoint_path": str(checkpoint_path),
        },
    )

    import torch  # type: ignore

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    exported_checkpoint_path = output_dirs["exported"] / "best.pt"
    exported_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, exported_checkpoint_path)

    split_metrics: dict[str, dict[str, Any]] = {}
    task_name = str(resolve_config(training_config["task_config"])["task_name"])
    source_config_path = training_config.get("_config_path")
    evaluation_splits = ["val"]
    if bool(training_config.get("evaluate_test_after_train", True)):
        evaluation_splits.append("test")
    for split_name in evaluation_splits:
        evaluation_payload = run_inference(
            model,
            loaders[split_name],
            device=device,
            criterion=criterion,
            progress_desc=f"Evaluating {split_name}",
            show_progress=show_progress,
        )
        metrics_payload = compute_classification_metrics(
            evaluation_payload["y_true"],
            evaluation_payload["y_pred"],
            evaluation_payload["probabilities"],
            class_names=class_names,
        )
        calibration_payload = compute_calibration(evaluation_payload["probabilities"], evaluation_payload["y_true"])
        save_metric_artifacts(
            evaluation_payload=evaluation_payload,
            metrics_payload=metrics_payload,
            calibration_payload=calibration_payload,
            class_names=class_names,
            output_dirs=output_dirs,
            split_name=split_name,
            task_name=task_name,
            source_config_path=source_config_path,
            checkpoint_path=exported_checkpoint_path,
        )
        save_confusion_matrix(
            evaluation_payload["y_true"],
            evaluation_payload["y_pred"],
            class_names=class_names,
            output_csv=output_dirs["confusion_matrices"] / f"{split_name}_confusion_matrix.csv",
            output_png=output_dirs["confusion_matrices"] / f"{split_name}_confusion_matrix.png",
        )
        write_experiment_report(
            experiment_name=experiment_name,
            split_name=split_name,
            metrics={**metrics_payload["metrics"], **calibration_payload},
            output_path=output_dirs["reports"] / f"{split_name}_summary.md",
        )
        split_metrics[split_name] = {**metrics_payload["metrics"], **calibration_payload}

    figure_bundle = None
    if bool(training_config.get("auto_generate_paper_figures", True)) and "test" in split_metrics:
        try:
            figure_bundle = generate_paper_figure_bundle(
                train_config=training_config,
                checkpoint_path=exported_checkpoint_path,
                output_root=Path(training_config.get("output_root", "outputs")) / "paper_figures",
                device=str(training_config.get("paper_figures_device", device)),
                xai_count=int(training_config.get("paper_figures_xai_count", 6)),
            )
        except Exception as exc:
            figure_bundle = {"error": str(exc)}

    return {
        "training": training_result,
        "splits": split_metrics,
        "checkpoint_path": str(checkpoint_path),
        "exported_checkpoint_path": str(exported_checkpoint_path),
        "figure_bundle": figure_bundle,
    }
