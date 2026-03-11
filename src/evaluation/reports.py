from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

from config_utils import resolve_config, write_json, write_text
from evaluation.metrics import flatten_classification_report
from experiment_utils import write_csv_rows


BASELINE_MODELS = {"alexnet", "resnet18", "vgg16"}


def save_curve_artifacts(curves: dict[str, Any], output_dirs: dict[str, Path], split_name: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore

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
    for base_row, target, pred, probs in zip(
        evaluation_payload["prediction_rows"],
        evaluation_payload["y_true"],
        evaluation_payload["y_pred"],
        evaluation_payload["probabilities"] if evaluation_payload["probabilities"] is not None else [],
        strict=False,
    ):
        row = {
            **base_row,
            "true_label": class_names[target],
            "pred_label": class_names[pred],
            "correct": bool(target == pred),
        }
        if probs is not None:
            for class_index, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = float(probs[class_index])
        prediction_rows.append(row)
    write_csv_rows(output_dirs["predictions"] / f"{split_name}_predictions.csv", prediction_rows)


def write_experiment_report(
    experiment_name: str,
    split_name: str,
    metrics: dict[str, Any],
    output_path: str | Path,
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
    write_text(output_path, "\n".join(lines))


def write_baseline_rollup_summaries(output_root: str | Path = "outputs") -> None:
    output_root = Path(output_root)
    report_root = output_root / "reports"
    rows = [row for row in collect_experiment_rows(output_root) if row["model"] in BASELINE_MODELS]
    if not rows:
        empty_text = "# Baseline Results Summary\n\n- No executed baseline test results are available yet."
        write_text(report_root / "baseline_results_summary.md", empty_text)
        write_text(report_root / "baseline_model_comparison_summary.md", empty_text.replace("Results", "Model Comparison"))
        write_text(report_root / "baseline_error_analysis_summary.md", "# Baseline Error Analysis Summary\n\n- No executed baseline error analysis is available yet.")
        write_text(report_root / "baseline_explainability_summary.md", "# Baseline Explainability Summary\n\n- No executed explainability runs are available yet.")
        write_text(report_root / "best_baseline_recommendation.md", "# Best Baseline Recommendation\n\n- No executed baseline ranking is available yet.")
        return

    pattern_rows = rank_rows([row for row in rows if row["task"] == "pattern3"])
    severity_rows = rank_rows([row for row in rows if row["task"] == "severity5"])

    write_text(report_root / "baseline_results_summary.md", build_results_summary(pattern_rows, severity_rows, "Baseline"))
    write_text(report_root / "baseline_model_comparison_summary.md", build_baseline_model_comparison_summary(pattern_rows, severity_rows))
    write_text(report_root / "baseline_error_analysis_summary.md", build_error_analysis_summary(rows, "Baseline"))
    write_text(report_root / "baseline_explainability_summary.md", build_explainability_summary(rows, output_root, "Baseline"))
    write_text(report_root / "best_baseline_recommendation.md", build_best_recommendation(pattern_rows, severity_rows))


def write_convnextv2_rollup_summaries(output_root: str | Path = "outputs") -> None:
    output_root = Path(output_root)
    report_root = output_root / "reports"
    rows = collect_experiment_rows(output_root)
    convnext_rows = [row for row in rows if row["model"].startswith("convnextv2")]
    baseline_rows = [row for row in rows if row["model"] in BASELINE_MODELS]
    baseline_pattern = rank_rows([row for row in baseline_rows if row["task"] == "pattern3"])
    baseline_reference = baseline_pattern[0] if baseline_pattern else None

    if not convnext_rows:
        empty_text = "# ConvNeXtV2 Results Summary\n\n- No executed ConvNeXtV2 runs are available yet."
        write_text(report_root / "convnextv2_results_summary.md", empty_text)
        write_text(report_root / "convnextv2_vs_baseline_summary.md", "# ConvNeXtV2 vs Baseline Summary\n\n- No executed ConvNeXtV2 runs are available yet.")
        write_text(report_root / "convnextv2_error_analysis_summary.md", "# ConvNeXtV2 Error Analysis Summary\n\n- No executed ConvNeXtV2 error analysis is available yet.")
        write_text(report_root / "convnextv2_explainability_summary.md", "# ConvNeXtV2 Explainability Summary\n\n- No executed ConvNeXtV2 explainability runs are available yet.")
        write_text(report_root / "final_model_recommendation.md", "# Final Model Recommendation\n\n- No executed ConvNeXtV2 comparison is available yet.")
        return

    pattern_rows = rank_rows([row for row in convnext_rows if row["task"] == "pattern3"])
    severity_rows = rank_rows([row for row in convnext_rows if row["task"] == "severity5"])

    write_text(report_root / "convnextv2_results_summary.md", build_results_summary(pattern_rows, severity_rows, "ConvNeXtV2"))
    write_text(report_root / "convnextv2_vs_baseline_summary.md", build_convnext_vs_baseline_summary(pattern_rows, baseline_reference))
    write_text(report_root / "convnextv2_error_analysis_summary.md", build_error_analysis_summary(convnext_rows, "ConvNeXtV2"))
    write_text(report_root / "convnextv2_explainability_summary.md", build_explainability_summary(convnext_rows, output_root, "ConvNeXtV2"))
    write_text(report_root / "final_model_recommendation.md", build_final_model_recommendation(pattern_rows, severity_rows, baseline_reference))


def collect_experiment_rows(output_root: Path) -> list[dict[str, Any]]:
    metric_root = output_root / "metrics"
    rows: list[dict[str, Any]] = []
    for metric_file in metric_root.glob("*/test_metrics.json"):
        experiment_name = metric_file.parent.name
        if "__holdout_v1__seed" not in experiment_name:
            continue
        task_slug, model_name, input_name, split_name, seed_name = parse_experiment_name(experiment_name)
        payload = json.loads(metric_file.read_text(encoding="utf-8"))
        val_metric_file = metric_file.parent / "val_metrics.json"
        val_payload = json.loads(val_metric_file.read_text(encoding="utf-8")) if val_metric_file.exists() else {}
        confusion_summary = summarize_confusion_matrix(
            output_root / "confusion_matrices" / experiment_name / "test_confusion_matrix.csv"
        )
        explainability_counts = summarize_explainability_dir(output_root / "explainability" / experiment_name)
        run_metadata = load_json_if_exists(output_root / "reports" / experiment_name / "run_metadata.json")
        training_summary = load_json_if_exists(metric_file.parent / "training_summary.json")
        resolved_config = {}
        config_path = run_metadata.get("config_path")
        if config_path:
            try:
                resolved_config = resolve_config(config_path)
            except Exception:
                resolved_config = {}
        checkpoint_path = training_summary.get("checkpoint_path")
        checkpoint_size_mb = None
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint_size_mb = round(Path(checkpoint_path).stat().st_size / (1024 * 1024), 2)
        rows.append(
            {
                "experiment_name": experiment_name,
                "task": task_slug,
                "model": model_name,
                "input": input_name,
                "split_name": split_name,
                "seed_name": seed_name,
                "balanced_accuracy": payload.get("balanced_accuracy"),
                "macro_f1": payload.get("macro_f1"),
                "weighted_f1": payload.get("weighted_f1"),
                "accuracy": payload.get("accuracy"),
                "val_balanced_accuracy": val_payload.get("balanced_accuracy"),
                "val_macro_f1": val_payload.get("macro_f1"),
                "stability_gap": stability_gap(val_payload.get("balanced_accuracy"), payload.get("balanced_accuracy")),
                "per_class_recall": recall_summary(payload.get("classification_report", {})),
                "confusion_summary": confusion_summary,
                "explainability_counts": explainability_counts,
                "config_path": config_path,
                "batch_size": resolved_config.get("batch_size"),
                "image_size": resolved_config.get("image_size"),
                "epochs": resolved_config.get("epochs"),
                "best_epoch": training_summary.get("best_epoch"),
                "checkpoint_size_mb": checkpoint_size_mb,
            }
        )
    return rows


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    rows.sort(
        key=lambda row: (
            row["balanced_accuracy"] if row["balanced_accuracy"] is not None else float("-inf"),
            row["macro_f1"] if row["macro_f1"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    return rows


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_experiment_name(experiment_name: str) -> tuple[str, str, str, str, str]:
    parts = experiment_name.split("__")
    if len(parts) >= 5:
        return parts[0], parts[1], parts[2], parts[3], parts[4]
    return experiment_name, "unknown", "unknown", "unknown", "unknown"


def stability_gap(val_score: Any, test_score: Any) -> float | None:
    if val_score is None or test_score is None:
        return None
    return abs(float(val_score) - float(test_score))


def recall_summary(report: dict[str, Any]) -> dict[str, float]:
    rows: dict[str, float] = {}
    for label, payload in report.items():
        if not isinstance(payload, dict):
            continue
        if label in {"macro avg", "weighted avg"}:
            continue
        rows[label] = float(payload.get("recall", 0.0))
    return rows


def summarize_confusion_matrix(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"top_confusions": [], "worst_recall_class": None}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if len(rows) <= 1:
        return {"top_confusions": [], "worst_recall_class": None}
    labels = rows[0][1:]
    matrix = []
    for row in rows[1:]:
        matrix.append([int(value) for value in row[1:]])
    confusions: list[tuple[int, str, str]] = []
    recalls: list[tuple[float, str]] = []
    for true_index, true_label in enumerate(labels):
        row_total = sum(matrix[true_index])
        diag = matrix[true_index][true_index]
        recall = (diag / row_total) if row_total else 0.0
        recalls.append((recall, true_label))
        for pred_index, pred_label in enumerate(labels):
            if pred_index == true_index:
                continue
            count = matrix[true_index][pred_index]
            if count > 0:
                confusions.append((count, true_label, pred_label))
    confusions.sort(reverse=True)
    recalls.sort()
    return {
        "top_confusions": [
            {"count": count, "true_label": true_label, "pred_label": pred_label}
            for count, true_label, pred_label in confusions[:3]
        ],
        "worst_recall_class": recalls[0][1] if recalls else None,
    }


def summarize_explainability_dir(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for bucket in ("correct", "incorrect", "borderline"):
        bucket_path = path / bucket
        counts[bucket] = len(list(bucket_path.glob("*.png"))) if bucket_path.exists() else 0
    return counts


def format_run_line(row: dict[str, Any]) -> str:
    return (
        f"- {row['experiment_name']}: balanced_accuracy={row['balanced_accuracy']}, "
        f"macro_f1={row['macro_f1']}, weighted_f1={row['weighted_f1']}, "
        f"val_test_gap={row['stability_gap']}"
    )


def build_results_summary(pattern_rows: list[dict[str, Any]], severity_rows: list[dict[str, Any]], family_name: str) -> str:
    lines = [
        f"# {family_name} Results Summary",
        "",
        "## Pattern 3-Class Holdout Benchmark",
        "",
    ]
    if pattern_rows:
        lines.extend(format_run_line(row) for row in pattern_rows)
    else:
        lines.append("- No completed pattern 3-class benchmark runs are available yet.")
    lines.extend(["", "## Severity 5-Class Extension", ""])
    if severity_rows:
        lines.extend(format_run_line(row) for row in severity_rows)
    else:
        lines.append("- No completed severity extension run is available yet.")
    return "\n".join(lines)


def build_baseline_model_comparison_summary(pattern_rows: list[dict[str, Any]], severity_rows: list[dict[str, Any]]) -> str:
    lines = ["# Baseline Model Comparison Summary", ""]
    if not pattern_rows:
        lines.append("- No completed pattern 3-class benchmark runs are available yet.")
        return "\n".join(lines)

    top_pattern = pattern_rows[0]
    most_stable = min(
        pattern_rows,
        key=lambda row: row["stability_gap"] if row["stability_gap"] is not None else float("inf"),
    )
    lines.append(f"- Best pattern run by ranking: {top_pattern['experiment_name']}")
    lines.append(f"- Most stable pattern run by smallest val/test balanced-accuracy gap: {most_stable['experiment_name']}")
    lines.append("")
    lines.append("## Raw vs Masked Comparison")
    lines.append("")
    for model_name in sorted({row["model"] for row in pattern_rows}):
        raw_row = next((row for row in pattern_rows if row["model"] == model_name and row["input"] == "raw_rgb"), None)
        masked_row = next(
            (row for row in pattern_rows if row["model"] == model_name and row["input"] == "masked_highlight_proxy"),
            None,
        )
        if raw_row and masked_row:
            delta = float(masked_row["balanced_accuracy"]) - float(raw_row["balanced_accuracy"])
            lines.append(
                f"- {model_name}: raw_rgb={raw_row['balanced_accuracy']} vs masked_highlight_proxy={masked_row['balanced_accuracy']} "
                f"(delta={delta:+.4f})"
            )
        elif raw_row:
            lines.append(f"- {model_name}: only raw_rgb run is available.")
        elif masked_row:
            lines.append(f"- {model_name}: only masked_highlight_proxy run is available.")
    if severity_rows:
        lines.extend(["", "## Severity Extension", "", f"- Executed severity run: {severity_rows[0]['experiment_name']}"])
    return "\n".join(lines)


def build_error_analysis_summary(rows: list[dict[str, Any]], family_name: str) -> str:
    lines = [f"# {family_name} Error Analysis Summary", ""]
    if not rows:
        lines.append(f"- No executed {family_name.lower()} error analysis is available yet.")
        return "\n".join(lines)
    for row in rank_rows(rows):
        lines.append(f"## {row['experiment_name']}")
        lines.append("")
        top_confusions = row["confusion_summary"]["top_confusions"]
        if top_confusions:
            for confusion in top_confusions:
                lines.append(
                    f"- Confusion: {confusion['true_label']} -> {confusion['pred_label']} count={confusion['count']}"
                )
        else:
            lines.append("- No off-diagonal confusions recorded.")
        worst_class = row["confusion_summary"].get("worst_recall_class")
        if worst_class:
            recall_value = row["per_class_recall"].get(worst_class)
            lines.append(f"- Lowest-recall class: {worst_class} recall={recall_value}")
        lines.append(
            "- Use the paired prediction CSV and confusion matrix to inspect whether these errors cluster around label ambiguity or minority-class collapse."
        )
        lines.append("")
    return "\n".join(lines)


def build_explainability_summary(rows: list[dict[str, Any]], output_root: Path, family_name: str) -> str:
    lines = [f"# {family_name} Explainability Summary", ""]
    if not rows:
        lines.append(f"- No executed {family_name.lower()} explainability runs are available yet.")
        return "\n".join(lines)
    for row in rank_rows(rows):
        explain_dir = output_root / "explainability" / row["experiment_name"]
        counts = row["explainability_counts"]
        lines.append(f"## {row['experiment_name']}")
        lines.append("")
        lines.append(f"- Explainability directory: `{explain_dir}`")
        lines.append(
            f"- Overlay counts: correct={counts.get('correct', 0)}, incorrect={counts.get('incorrect', 0)}, borderline={counts.get('borderline', 0)}"
        )
        lines.append("")
    return "\n".join(lines)


def build_best_recommendation(pattern_rows: list[dict[str, Any]], severity_rows: list[dict[str, Any]]) -> str:
    lines = ["# Best Baseline Recommendation", ""]
    if not pattern_rows:
        lines.append("- No executed baseline ranking is available yet.")
        return "\n".join(lines)
    best = pattern_rows[0]
    most_stable = min(
        pattern_rows,
        key=lambda row: row["stability_gap"] if row["stability_gap"] is not None else float("inf"),
    )
    lines.append(f"- Official Stage 3 reference baseline: {best['experiment_name']}")
    lines.append(f"- Balanced accuracy: {best['balanced_accuracy']}")
    lines.append(f"- Macro F1: {best['macro_f1']}")
    lines.append(f"- Most stable completed pattern baseline: {most_stable['experiment_name']}")
    if best["model"] == most_stable["model"] and best["input"] == most_stable["input"]:
        lines.append("- The top-scoring and most-stable completed pattern baseline are the same configuration.")
    else:
        lines.append("- The top-scoring and most-stable completed pattern baselines differ; carry both into interpretation.")
    if severity_rows:
        lines.append(f"- Severity extension executed with: {severity_rows[0]['experiment_name']}")
    lines.append("- Use balanced accuracy, macro F1, and per-class recall as the ranking basis, not plain accuracy.")
    lines.append("- These recommendations summarize computed project results only.")
    return "\n".join(lines)


def build_convnext_vs_baseline_summary(pattern_rows: list[dict[str, Any]], baseline_reference: dict[str, Any] | None) -> str:
    lines = ["# ConvNeXtV2 vs Baseline Summary", ""]
    if not pattern_rows:
        lines.append("- No completed ConvNeXtV2 pattern runs are available yet.")
        return "\n".join(lines)
    best = pattern_rows[0]
    lines.append(f"- Best ConvNeXtV2 pattern run by ranking: {best['experiment_name']}")
    if baseline_reference:
        bal_delta = float(best["balanced_accuracy"]) - float(baseline_reference["balanced_accuracy"])
        macro_delta = float(best["macro_f1"]) - float(baseline_reference["macro_f1"])
        lines.append(f"- Official Stage 3 baseline reference: {baseline_reference['experiment_name']}")
        lines.append(f"- Balanced accuracy delta vs baseline: {bal_delta:+.4f}")
        lines.append(f"- Macro F1 delta vs baseline: {macro_delta:+.4f}")
        lines.append(
            f"- Baseline reference metrics: balanced_accuracy={baseline_reference['balanced_accuracy']}, macro_f1={baseline_reference['macro_f1']}"
        )
    lines.extend(["", "## Raw vs Masked Comparison", ""])
    for model_name in sorted({row["model"] for row in pattern_rows}):
        raw_row = next((row for row in pattern_rows if row["model"] == model_name and row["input"] == "raw_rgb"), None)
        masked_row = next(
            (row for row in pattern_rows if row["model"] == model_name and row["input"] == "masked_highlight_proxy"),
            None,
        )
        if raw_row and masked_row:
            delta = float(masked_row["balanced_accuracy"]) - float(raw_row["balanced_accuracy"])
            lines.append(
                f"- {model_name}: raw_rgb={raw_row['balanced_accuracy']} vs masked_highlight_proxy={masked_row['balanced_accuracy']} "
                f"(delta={delta:+.4f})"
            )
        elif raw_row:
            lines.append(f"- {model_name}: only raw_rgb run is available.")
        elif masked_row:
            lines.append(f"- {model_name}: only masked_highlight_proxy run is available.")
    tiny_raw = next((row for row in pattern_rows if row["model"] == "convnextv2_tiny" and row["input"] == "raw_rgb"), None)
    stronger_raw_candidates = [
        row for row in pattern_rows if row["model"] != "convnextv2_tiny" and row["input"] == "raw_rgb"
    ]
    stronger_raw = rank_rows(stronger_raw_candidates)[0] if stronger_raw_candidates else None
    lines.extend(["", "## Stronger Variant Check", ""])
    if tiny_raw and stronger_raw:
        bal_delta = float(stronger_raw["balanced_accuracy"]) - float(tiny_raw["balanced_accuracy"])
        macro_delta = float(stronger_raw["macro_f1"]) - float(tiny_raw["macro_f1"])
        lines.append(
            f"- Stronger variant comparison: {stronger_raw['experiment_name']} vs {tiny_raw['experiment_name']}"
        )
        lines.append(f"- Balanced accuracy delta: {bal_delta:+.4f}")
        lines.append(f"- Macro F1 delta: {macro_delta:+.4f}")
        lines.append(
            f"- Cost proxy: stronger checkpoint={stronger_raw.get('checkpoint_size_mb')} MB batch_size={stronger_raw.get('batch_size')} "
            f"vs tiny checkpoint={tiny_raw.get('checkpoint_size_mb')} MB batch_size={tiny_raw.get('batch_size')}"
        )
    elif tiny_raw:
        lines.append("- No stronger ConvNeXtV2 raw run is available yet.")
    else:
        lines.append("- No ConvNeXtV2 tiny raw run is available yet.")
    return "\n".join(lines)


def build_final_model_recommendation(
    pattern_rows: list[dict[str, Any]],
    severity_rows: list[dict[str, Any]],
    baseline_reference: dict[str, Any] | None,
) -> str:
    lines = ["# Final Model Recommendation", ""]
    if not pattern_rows:
        lines.append("- No executed ConvNeXtV2 comparison is available yet.")
        return "\n".join(lines)
    best = pattern_rows[0]
    lines.append(f"- Best ConvNeXtV2 pattern run: {best['experiment_name']}")
    lines.append(f"- Balanced accuracy: {best['balanced_accuracy']}")
    lines.append(f"- Macro F1: {best['macro_f1']}")
    if baseline_reference:
        bal_delta = float(best["balanced_accuracy"]) - float(baseline_reference["balanced_accuracy"])
        macro_delta = float(best["macro_f1"]) - float(baseline_reference["macro_f1"])
        lines.append(f"- Baseline reference: {baseline_reference['experiment_name']}")
        lines.append(f"- Balanced accuracy delta vs baseline: {bal_delta:+.4f}")
        lines.append(f"- Macro F1 delta vs baseline: {macro_delta:+.4f}")
        if bal_delta > 0 and macro_delta >= 0:
            lines.append("- Recommendation: promote ConvNeXtV2 as the official final model family for the pattern 3-class task.")
        else:
            lines.append("- Recommendation: keep the Stage 3 AlexNet raw baseline as the official reference and treat ConvNeXtV2 as exploratory or task-specific.")
    else:
        lines.append("- No official baseline reference was available for direct comparison.")
    if severity_rows:
        lines.append(f"- Severity extension executed with: {severity_rows[0]['experiment_name']}")
    lines.append("- Raw RGB remains the default path unless a matched masked_highlight_proxy run shows a real holdout gain.")
    lines.append("- Ranking is based on balanced accuracy first, then macro F1, then per-class recall review.")
    lines.append("- These recommendations summarize computed project results only.")
    return "\n".join(lines)
