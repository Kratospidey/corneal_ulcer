from __future__ import annotations

from pathlib import Path
from typing import Any

from config_utils import write_json, write_text
from evaluation.metrics import flatten_classification_report
from evaluation.prediction_contract import (
    build_prediction_provenance,
    build_prediction_row,
    logit_column_names,
    validate_prediction_provenance,
    validate_prediction_rows,
)
from experiment_utils import write_csv_rows

def save_curve_artifacts(curves: dict[str, Any], output_dirs: dict[str, Path], split_name: str) -> None:
    import matplotlib  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    matplotlib.use("Agg")
    for curve_name, curve_payload in (("roc", curves.get("roc", {})), ("pr", curves.get("pr", {}))):
        if not curve_payload:
            continue
        figure_dir = output_dirs["roc_curves"] if curve_name == "roc" else output_dirs["pr_curves"]
        for class_name, values in curve_payload.items():
            fig, axis = plt.subplots(figsize=(6, 5))
            if curve_name == "roc":
                axis.plot(values["fpr"], values["tpr"])
                axis.set_xlabel("False positive rate")
                axis.set_ylabel("True positive rate")
                axis.set_title(f"{split_name.title()} ROC: {class_name}")
            else:
                axis.plot(values["recall"], values["precision"])
                axis.set_xlabel("Recall")
                axis.set_ylabel("Precision")
                axis.set_title(f"{split_name.title()} PR: {class_name}")
            fig.tight_layout()
            fig.savefig(figure_dir / f"{split_name}_{curve_name}_{class_name}.png")
            plt.close(fig)


def save_metric_artifacts(
    evaluation_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
    calibration_payload: dict[str, Any],
    class_names: list[str] | tuple[str, ...],
    output_dirs: dict[str, Path],
    split_name: str,
    task_name: str | None = None,
    source_config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> None:
    metrics = dict(metrics_payload["metrics"])
    if evaluation_payload["loss"] is not None:
        metrics["loss"] = evaluation_payload["loss"]
    metrics.update(calibration_payload)

    write_json(output_dirs["metrics"] / f"{split_name}_metrics.json", metrics)
    write_csv_rows(
        output_dirs["metrics"] / f"{split_name}_classification_report.csv",
        flatten_classification_report(metrics["classification_report"]),
    )
    save_curve_artifacts(metrics_payload["curves"], output_dirs, split_name)

    prediction_rows = []
    logit_matrix = evaluation_payload.get("logits")
    for base_row, target, pred, probs in zip(
        evaluation_payload["prediction_rows"],
        evaluation_payload["y_true"],
        evaluation_payload["y_pred"],
        evaluation_payload["probabilities"] if evaluation_payload["probabilities"] is not None else [],
        strict=False,
    ):
        prediction_rows.append(
            build_prediction_row(
                base_row=base_row,
                class_names=class_names,
                probabilities=[float(value) for value in probs.tolist()],
                logits=[float(value) for value in (base_row.get("logits") or [])] if base_row.get("logits") else None,
                extras={
                    "confidence": base_row.get("confidence"),
                    "raw_image_path": base_row.get("raw_image_path", ""),
                    "cornea_mask_path": base_row.get("cornea_mask_path", ""),
                    "ulcer_mask_path": base_row.get("ulcer_mask_path", ""),
                    "true_label": class_names[target],
                    "pred_label": class_names[pred],
                    "correct": bool(target == pred),
                },
            )
        )
    validate_prediction_rows(prediction_rows, class_names=class_names, split_name=split_name)
    write_csv_rows(output_dirs["predictions"] / f"{split_name}_predictions.csv", prediction_rows)
    if task_name is not None:
        prediction_provenance = build_prediction_provenance(
            task_name=task_name,
            class_names=class_names,
            split_name=split_name,
            source_config_path=source_config_path,
            checkpoint_path=checkpoint_path,
            include_logits=bool(logit_matrix is not None),
        )
        validate_prediction_provenance(
            prediction_provenance,
            task_name=task_name,
            class_names=class_names,
            split_name=split_name,
        )
        write_json(output_dirs["predictions"] / f"{split_name}_prediction_metadata.json", prediction_provenance)


def write_experiment_report(
    experiment_name: str,
    split_name: str,
    metrics: dict[str, Any],
    output_path: str | Path,
    report_context: dict[str, Any] | None = None,
) -> None:
    lines = [
        f"# {experiment_name} {split_name.title()} Summary",
        "",
        f"- Accuracy: {metrics.get('accuracy')}",
        f"- Balanced accuracy: {metrics.get('balanced_accuracy')}",
        f"- Macro F1: {metrics.get('macro_f1')}",
        f"- Weighted F1: {metrics.get('weighted_f1')}",
        f"- ROC-AUC macro OVR: {metrics.get('roc_auc_macro_ovr')}",
        f"- PR-AUC macro OVR: {metrics.get('pr_auc_macro_ovr')}",
        f"- ECE: {metrics.get('ece')}",
        "",
        "## Interpretation",
        "",
        "- Balanced accuracy, macro F1, and per-class recall should be treated as the primary medical-classification signals.",
        "- This report summarizes computed project metrics only.",
    ]
    if report_context:
        context_lines = []
        if report_context.get("artifact_path") is not None:
            context_lines.append(f"- Artifact path: {report_context['artifact_path']}")
        if report_context.get("split_file") is not None:
            context_lines.append(f"- Split file: {report_context['split_file']}")
        if report_context.get("seed") is not None:
            context_lines.append(f"- Seed: {report_context['seed']}")
        if context_lines:
            lines.extend(["", "## Context", "", *context_lines])
    write_text(output_path, "\n".join(lines))
