from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from experimental.severity.train_factorized_tabular import (
    STAGE_CLASS_NAMES,
    feature_columns as stage_feature_columns,
    predict_s0_rules,
    predict_s1_rules,
    predict_s2_rules,
)
from utils_io import write_csv_rows, write_json, write_text


FULL_CLASS_NAMES = (
    "no_ulcer",
    "ulcer_leq_25pct",
    "ulcer_leq_50pct",
    "ulcer_geq_75pct",
    "central_ulcer",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(FULL_CLASS_NAMES)}
STRICT_REFERENCE = {
    "name": "severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_tempered_v1__holdout_v1__seed42",
    "balanced_accuracy": 0.6108918987988756,
    "macro_f1": 0.55424317617866,
}
HGB_FALLBACK = {
    "name": "hgb_fallback",
    "balanced_accuracy": 0.3279649033137405,
    "macro_f1": 0.3327291324180098,
}
BEST_SEV_S1 = {
    "name": "severity5__posthoc__geom_plus_patternlogits_hgb_v1__holdout_v1",
    "balanced_accuracy": 0.39928102904847085,
    "macro_f1": 0.401951611606784,
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate combined SEV-S2 factorized stage predictions.")
    parser.add_argument("--geom-table", required=True)
    parser.add_argument("--s0-experiment", required=True)
    parser.add_argument("--s1-experiment", required=True)
    parser.add_argument("--s2-experiment", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--report-path")
    return parser


def adjacent_error_rate(y_true: list[str], y_pred: list[str]) -> float:
    adjacent_errors = 0
    all_errors = 0
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            continue
        all_errors += 1
        if abs(CLASS_TO_INDEX[true_label] - CLASS_TO_INDEX[pred_label]) <= 1:
            adjacent_errors += 1
    return float(adjacent_errors / max(all_errors, 1))


def compute_full_metrics(y_true: list[str], y_pred: list[str], probabilities: np.ndarray) -> dict[str, object]:
    report = classification_report(y_true, y_pred, labels=list(FULL_CLASS_NAMES), output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_true, y_pred, labels=list(FULL_CLASS_NAMES)).tolist()
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(FULL_CLASS_NAMES), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(FULL_CLASS_NAMES), average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion,
        "central_ulcer_recall": float(report["central_ulcer"]["recall"]),
        "no_ulcer_precision": float(report["no_ulcer"]["precision"]),
        "adjacent_class_error_rate": adjacent_error_rate(y_true, y_pred),
        "mean_confidence": float(np.max(probabilities, axis=1).mean()),
    }


def load_training_summary(output_root: Path, experiment_name: str) -> dict[str, object]:
    path = output_root / "reports" / experiment_name / "training_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))

def stage_predict(frame: pd.DataFrame, output_root: Path, experiment_name: str) -> tuple[list[str], np.ndarray]:
    summary = load_training_summary(output_root, experiment_name)
    stage_name = str(summary["stage_name"])
    class_names = tuple(summary["class_names"])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    feature_mode = str(summary.get("feature_mode", "all_numeric"))
    features = stage_feature_columns(frame, feature_mode)
    model_name = str(summary["model"])

    if model_name == "rules":
        thresholds = dict(summary["rule_thresholds"])
        if stage_name == "s0":
            return predict_s0_rules(frame, thresholds, class_to_index)
        if stage_name == "s1":
            return predict_s1_rules(frame, thresholds, class_to_index)
        return predict_s2_rules(frame, thresholds, class_to_index)

    model_path = output_root / "debug" / "severity_posthoc" / experiment_name / "model.joblib"
    model = joblib.load(model_path)
    probabilities = model.predict_proba(frame[features].to_numpy(dtype=np.float32))
    predictions = [class_names[index] for index in np.argmax(probabilities, axis=1)]
    return predictions, probabilities


def combine_split(
    base_df: pd.DataFrame,
    output_root: Path,
    s0_experiment: str,
    s1_experiment: str,
    s2_experiment: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    s0_predictions, s0_probabilities = stage_predict(base_df, output_root, s0_experiment)
    s1_predictions, s1_probabilities = stage_predict(base_df, output_root, s1_experiment)
    s2_predictions, s2_probabilities = stage_predict(base_df, output_root, s2_experiment)

    s0_index = {name: index for index, name in enumerate(STAGE_CLASS_NAMES["s0"])}
    s1_index = {name: index for index, name in enumerate(STAGE_CLASS_NAMES["s1"])}
    s2_index = {name: index for index, name in enumerate(STAGE_CLASS_NAMES["s2"])}

    final_predictions: list[str] = []
    final_probabilities = np.zeros((len(base_df), len(FULL_CLASS_NAMES)), dtype=np.float32)
    rows: list[dict[str, object]] = []
    for index, (_, row) in enumerate(base_df.iterrows()):
        s0_pred = s0_predictions[index]
        s1_pred = s1_predictions[index]
        s2_pred = s2_predictions[index]

        p_no_ulcer = float(s0_probabilities[index][s0_index["no_ulcer"]])
        p_ulcer_present = float(s0_probabilities[index][s0_index["ulcer_present"]])
        p_central = float(s1_probabilities[index][s1_index["central_ulcer"]])
        p_noncentral = float(s1_probabilities[index][s1_index["noncentral_ulcer"]])
        p_leq25 = float(s2_probabilities[index][s2_index["ulcer_leq_25pct"]])
        p_leq50 = float(s2_probabilities[index][s2_index["ulcer_leq_50pct"]])
        p_geq75 = float(s2_probabilities[index][s2_index["ulcer_geq_75pct"]])

        final_probabilities[index, CLASS_TO_INDEX["no_ulcer"]] = p_no_ulcer
        final_probabilities[index, CLASS_TO_INDEX["central_ulcer"]] = p_ulcer_present * p_central
        final_probabilities[index, CLASS_TO_INDEX["ulcer_leq_25pct"]] = p_ulcer_present * p_noncentral * p_leq25
        final_probabilities[index, CLASS_TO_INDEX["ulcer_leq_50pct"]] = p_ulcer_present * p_noncentral * p_leq50
        final_probabilities[index, CLASS_TO_INDEX["ulcer_geq_75pct"]] = p_ulcer_present * p_noncentral * p_geq75

        if s0_pred == "no_ulcer":
            final_label = "no_ulcer"
            route = "s0:no_ulcer"
        elif s1_pred == "central_ulcer":
            final_label = "central_ulcer"
            route = "s0:ulcer_present>s1:central_ulcer"
        else:
            final_label = s2_pred
            route = f"s0:ulcer_present>s1:noncentral_ulcer>s2:{s2_pred}"

        final_predictions.append(final_label)
        rows.append(
            {
                "image_id": str(row["image_id"]),
                "split": str(row["split"]),
                "true_label": str(row["severity_label"]),
                "pred_label": final_label,
                "route_taken": route,
                "s0_pred": s0_pred,
                "s1_pred": s1_pred,
                "s2_pred": s2_pred,
                "correct": bool(row["severity_label"] == final_label),
                "confidence": float(np.max(final_probabilities[index])),
                "raw_image_path": str(row["raw_image_path"]),
            }
        )
    metrics = compute_full_metrics(base_df["severity_label"].tolist(), final_predictions, final_probabilities)
    return pd.DataFrame(rows), metrics


def save_outputs(output_root: Path, experiment_name: str, split_name: str, rows: pd.DataFrame, metrics: dict[str, object]) -> None:
    metrics_dir = output_root / "metrics" / experiment_name
    predictions_dir = output_root / "predictions" / experiment_name
    reports_dir = output_root / "reports" / experiment_name
    debug_dir = output_root / "debug" / "severity_posthoc" / experiment_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / f"{split_name}_metrics.json", metrics)
    write_csv_rows(predictions_dir / f"{split_name}_predictions.csv", rows.to_dict(orient="records"))
    confusion_rows = []
    for true_index, true_label in enumerate(FULL_CLASS_NAMES):
        payload = {"true_label": true_label}
        for pred_index, pred_label in enumerate(FULL_CLASS_NAMES):
            payload[pred_label] = int(metrics["confusion_matrix"][true_index][pred_index])
        confusion_rows.append(payload)
    write_csv_rows(debug_dir / f"{split_name}_confusion_matrix.csv", confusion_rows)


def mild_bin_diagnostics(predictions: pd.DataFrame) -> dict[str, object]:
    mild = predictions[predictions["true_label"].isin(("ulcer_leq_25pct", "ulcer_leq_50pct"))]
    no_ulcer_rows = predictions[predictions["true_label"] == "no_ulcer"]
    return {
        "ulcer_leq_25pct_pred_breakdown": mild[mild["true_label"] == "ulcer_leq_25pct"]["pred_label"].value_counts().to_dict(),
        "ulcer_leq_50pct_pred_breakdown": mild[mild["true_label"] == "ulcer_leq_50pct"]["pred_label"].value_counts().to_dict(),
        "no_ulcer_false_negatives": int(((predictions["true_label"] == "no_ulcer") & (predictions["pred_label"] != "no_ulcer")).sum()),
        "no_ulcer_false_positives": int(((predictions["true_label"] != "no_ulcer") & (predictions["pred_label"] == "no_ulcer")).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    geom_table = pd.read_csv(args.geom_table)
    geom_table["image_id"] = geom_table["image_id"].astype(str)

    comparison_rows = []
    final_report_lines = [
        "# SEV-S2 Factorized Post-Hoc Results",
        "",
        f"- Strict reference: `{STRICT_REFERENCE['name']}` BA {STRICT_REFERENCE['balanced_accuracy']:.4f}, macro F1 {STRICT_REFERENCE['macro_f1']:.4f}",
        f"- SEV-S1 best: `{BEST_SEV_S1['name']}` BA {BEST_SEV_S1['balanced_accuracy']:.4f}, macro F1 {BEST_SEV_S1['macro_f1']:.4f}",
        f"- Fallback: `{HGB_FALLBACK['name']}` BA {HGB_FALLBACK['balanced_accuracy']:.4f}, macro F1 {HGB_FALLBACK['macro_f1']:.4f}",
        "",
    ]

    split_metrics: dict[str, dict[str, object]] = {}
    stage_metrics = {
        "s0": json.loads((output_root / "metrics" / args.s0_experiment / "test_metrics.json").read_text(encoding="utf-8")),
        "s1": json.loads((output_root / "metrics" / args.s1_experiment / "test_metrics.json").read_text(encoding="utf-8")),
        "s2": json.loads((output_root / "metrics" / args.s2_experiment / "test_metrics.json").read_text(encoding="utf-8")),
    }

    for split_name in ("val", "test"):
        base_df = geom_table[geom_table["split"] == split_name].reset_index(drop=True)
        combined_rows, metrics = combine_split(base_df, output_root, args.s0_experiment, args.s1_experiment, args.s2_experiment)
        split_metrics[split_name] = metrics
        save_outputs(output_root, args.experiment_name, split_name, combined_rows, metrics)
        if split_name == "test":
            diagnostics = mild_bin_diagnostics(combined_rows)
            write_json(output_root / "debug" / "severity_posthoc" / args.experiment_name / "test_error_diagnostics.json", diagnostics)

    test_metrics = split_metrics["test"]
    comparison_rows.append(
        {
            "experiment_name": args.experiment_name,
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "central_ulcer_recall": float(test_metrics["central_ulcer_recall"]),
            "no_ulcer_precision": float(test_metrics["no_ulcer_precision"]),
            "adjacent_class_error_rate": float(test_metrics["adjacent_class_error_rate"]),
            "delta_vs_hgb_fallback_ba": float(test_metrics["balanced_accuracy"]) - HGB_FALLBACK["balanced_accuracy"],
            "delta_vs_hgb_fallback_macro_f1": float(test_metrics["macro_f1"]) - HGB_FALLBACK["macro_f1"],
            "delta_vs_sev_s1_best_ba": float(test_metrics["balanced_accuracy"]) - BEST_SEV_S1["balanced_accuracy"],
            "delta_vs_sev_s1_best_macro_f1": float(test_metrics["macro_f1"]) - BEST_SEV_S1["macro_f1"],
            "delta_vs_strict_reference_ba": float(test_metrics["balanced_accuracy"]) - STRICT_REFERENCE["balanced_accuracy"],
            "delta_vs_strict_reference_macro_f1": float(test_metrics["macro_f1"]) - STRICT_REFERENCE["macro_f1"],
        }
    )
    write_csv_rows(output_root / "debug" / "severity_posthoc" / args.experiment_name / "comparison_summary.csv", comparison_rows)
    write_json(output_root / "reports" / args.experiment_name / "stage_test_metrics.json", stage_metrics)

    final_report_lines.extend(
        [
            "## Stage Test Metrics",
            "",
            f"- S0 `{args.s0_experiment}`: BA `{stage_metrics['s0']['balanced_accuracy']:.4f}`, macro F1 `{stage_metrics['s0']['macro_f1']:.4f}`",
            f"- S1 `{args.s1_experiment}`: BA `{stage_metrics['s1']['balanced_accuracy']:.4f}`, macro F1 `{stage_metrics['s1']['macro_f1']:.4f}`",
            f"- S2 `{args.s2_experiment}`: BA `{stage_metrics['s2']['balanced_accuracy']:.4f}`, macro F1 `{stage_metrics['s2']['macro_f1']:.4f}`",
            "",
            "## Combined Test Metrics",
            "",
            f"- balanced accuracy `{test_metrics['balanced_accuracy']:.4f}`",
            f"- macro F1 `{test_metrics['macro_f1']:.4f}`",
            f"- central-ulcer recall `{test_metrics['central_ulcer_recall']:.4f}`",
            f"- no-ulcer precision `{test_metrics['no_ulcer_precision']:.4f}`",
            f"- adjacent-class error rate `{test_metrics['adjacent_class_error_rate']:.4f}`",
            "",
            "## Comparison",
            "",
            f"- vs fallback: dBA `{comparison_rows[0]['delta_vs_hgb_fallback_ba']:.4f}`, dMacro F1 `{comparison_rows[0]['delta_vs_hgb_fallback_macro_f1']:.4f}`",
            f"- vs SEV-S1 best: dBA `{comparison_rows[0]['delta_vs_sev_s1_best_ba']:.4f}`, dMacro F1 `{comparison_rows[0]['delta_vs_sev_s1_best_macro_f1']:.4f}`",
            f"- vs strict reference: dBA `{comparison_rows[0]['delta_vs_strict_reference_ba']:.4f}`, dMacro F1 `{comparison_rows[0]['delta_vs_strict_reference_macro_f1']:.4f}`",
            "",
            "## Per-Class Recall / F1",
            "",
        ]
    )
    report = test_metrics["classification_report"]
    for class_name in FULL_CLASS_NAMES:
        final_report_lines.append(
            f"- `{class_name}`: `{float(report[class_name]['recall']):.4f} / {float(report[class_name]['f1-score']):.4f}`"
        )
    final_report_lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            f"- labels: `{', '.join(FULL_CLASS_NAMES)}`",
            f"- matrix: `{test_metrics['confusion_matrix']}`",
        ]
    )
    report_text = "\n".join(final_report_lines)
    write_text(output_root / "reports" / args.experiment_name / "test_summary.md", report_text)
    if args.report_path:
        write_text(args.report_path, report_text)
    print(json.dumps({"experiment_name": args.experiment_name, "test_balanced_accuracy": test_metrics["balanced_accuracy"], "test_macro_f1": test_metrics["macro_f1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
