from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from experimental.severity.eval_factorized_severity import (
    BEST_SEV_S1,
    CLASS_TO_INDEX,
    FULL_CLASS_NAMES,
    HGB_FALLBACK,
    STRICT_REFERENCE,
    compute_full_metrics,
    save_outputs,
)
from experimental.severity.train_factorized_tabular import (
    STAGE_CLASS_NAMES as FACTORIZED_STAGE_CLASS_NAMES,
    feature_columns as factorized_feature_columns,
    predict_s0_rules,
    predict_s1_rules,
)
from experimental.severity.train_s2_ordinal_tabular import (
    STAGE_CLASS_NAMES as S2_STAGE_CLASS_NAMES,
    predict_flat_rules,
    predict_geq75_rules,
    predict_leq25_rules,
)
from utils_io import write_csv_rows, write_json, write_text


BEST_SEV_S2 = {
    "name": "severity5__posthoc__factorized_geom_plus_patternlogits_hgb_v1__holdout_v1",
    "balanced_accuracy": 0.42129653292443986,
    "macro_f1": 0.4347415784168825,
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate SEV-S3 flat and ordinal S2 replacements on top of fixed SEV-S2 S0/S1.")
    parser.add_argument("--geom-table", required=True)
    parser.add_argument("--s0-experiment", required=True)
    parser.add_argument("--s1-experiment", required=True)
    parser.add_argument("--s2-flat-experiment", required=True)
    parser.add_argument("--s2-leq25-experiment", required=True)
    parser.add_argument("--s2-geq75-experiment", required=True)
    parser.add_argument("--flat-experiment-name", required=True)
    parser.add_argument("--ordinal-experiment-name", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--report-path")
    return parser


def load_training_summary(output_root: Path, experiment_name: str) -> dict[str, object]:
    path = output_root / "reports" / experiment_name / "training_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def stage_predict(frame: pd.DataFrame, output_root: Path, experiment_name: str) -> tuple[list[str], np.ndarray]:
    summary = load_training_summary(output_root, experiment_name)
    stage_name = str(summary["stage_name"])
    class_names = tuple(summary["class_names"])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    feature_mode = str(summary.get("feature_mode", "all_numeric"))
    model_name = str(summary["model"])
    saved_columns = [str(column) for column in summary.get("feature_columns", [])]

    if stage_name in FACTORIZED_STAGE_CLASS_NAMES:
        features = saved_columns or factorized_feature_columns(frame, feature_mode)
        if model_name == "rules":
            thresholds = dict(summary["rule_thresholds"])
            if stage_name == "s0":
                return predict_s0_rules(frame, thresholds, class_to_index)
            return predict_s1_rules(frame, thresholds, class_to_index)
        model = joblib.load(output_root / "debug" / "severity_posthoc" / experiment_name / "model.joblib")
        probabilities = model.predict_proba(frame[features].to_numpy(dtype=np.float32))
        predictions = [class_names[index] for index in np.argmax(probabilities, axis=1)]
        return predictions, probabilities

    features = saved_columns or factorized_feature_columns(frame, feature_mode)
    if model_name == "rules":
        thresholds = dict(summary["rule_thresholds"])
        if stage_name == "s2_flat":
            return predict_flat_rules(frame, thresholds, class_to_index)
        if stage_name == "s2_leq25":
            return predict_leq25_rules(frame, thresholds, class_to_index)
        return predict_geq75_rules(frame, thresholds, class_to_index)
    model = joblib.load(output_root / "debug" / "severity_posthoc" / experiment_name / "model.joblib")
    probabilities = model.predict_proba(frame[features].to_numpy(dtype=np.float32))
    predictions = [class_names[index] for index in np.argmax(probabilities, axis=1)]
    return predictions, probabilities


def combine_flat(
    base_df: pd.DataFrame,
    output_root: Path,
    s0_experiment: str,
    s1_experiment: str,
    s2_flat_experiment: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    s0_predictions, s0_probabilities = stage_predict(base_df, output_root, s0_experiment)
    s1_predictions, s1_probabilities = stage_predict(base_df, output_root, s1_experiment)
    s2_predictions, s2_probabilities = stage_predict(base_df, output_root, s2_flat_experiment)

    s0_index = {name: index for index, name in enumerate(FACTORIZED_STAGE_CLASS_NAMES["s0"])}
    s1_index = {name: index for index, name in enumerate(FACTORIZED_STAGE_CLASS_NAMES["s1"])}
    s2_index = {name: index for index, name in enumerate(S2_STAGE_CLASS_NAMES["s2_flat"])}

    rows: list[dict[str, object]] = []
    final_predictions: list[str] = []
    final_probabilities = np.zeros((len(base_df), len(FULL_CLASS_NAMES)), dtype=np.float32)

    for index, (_, row) in enumerate(base_df.iterrows()):
        s0_pred = s0_predictions[index]
        s1_pred = s1_predictions[index]
        s2_pred = s2_predictions[index]

        p_no_ulcer = float(s0_probabilities[index][s0_index["no_ulcer"]])
        p_ulcer_present = float(s0_probabilities[index][s0_index["ulcer_present"]])
        p_central = float(s1_probabilities[index][s1_index["central_ulcer"]])
        p_noncentral = float(s1_probabilities[index][s1_index["noncentral_ulcer"]])
        final_probabilities[index, CLASS_TO_INDEX["no_ulcer"]] = p_no_ulcer
        final_probabilities[index, CLASS_TO_INDEX["central_ulcer"]] = p_ulcer_present * p_central
        for label in S2_STAGE_CLASS_NAMES["s2_flat"]:
            final_probabilities[index, CLASS_TO_INDEX[label]] = p_ulcer_present * p_noncentral * float(s2_probabilities[index][s2_index[label]])

        if s0_pred == "no_ulcer":
            final_label = "no_ulcer"
            route = "s0:no_ulcer"
        elif s1_pred == "central_ulcer":
            final_label = "central_ulcer"
            route = "s0:ulcer_present>s1:central_ulcer"
        else:
            final_label = s2_pred
            route = f"s0:ulcer_present>s1:noncentral_ulcer>s2flat:{s2_pred}"
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
                "s2_flat_pred": s2_pred,
                "correct": bool(row["severity_label"] == final_label),
                "confidence": float(np.max(final_probabilities[index])),
                "raw_image_path": str(row["raw_image_path"]),
            }
        )
    return pd.DataFrame(rows), compute_full_metrics(base_df["severity_label"].tolist(), final_predictions, final_probabilities)


def combine_ordinal(
    base_df: pd.DataFrame,
    output_root: Path,
    s0_experiment: str,
    s1_experiment: str,
    s2_leq25_experiment: str,
    s2_geq75_experiment: str,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    s0_predictions, s0_probabilities = stage_predict(base_df, output_root, s0_experiment)
    s1_predictions, s1_probabilities = stage_predict(base_df, output_root, s1_experiment)
    b_predictions, b_probabilities = stage_predict(base_df, output_root, s2_leq25_experiment)
    c_predictions, c_probabilities = stage_predict(base_df, output_root, s2_geq75_experiment)

    s0_index = {name: index for index, name in enumerate(FACTORIZED_STAGE_CLASS_NAMES["s0"])}
    s1_index = {name: index for index, name in enumerate(FACTORIZED_STAGE_CLASS_NAMES["s1"])}
    b_index = {name: index for index, name in enumerate(S2_STAGE_CLASS_NAMES["s2_leq25"])}
    c_index = {name: index for index, name in enumerate(S2_STAGE_CLASS_NAMES["s2_geq75"])}

    rows: list[dict[str, object]] = []
    final_predictions: list[str] = []
    final_probabilities = np.zeros((len(base_df), len(FULL_CLASS_NAMES)), dtype=np.float32)
    conflicts = 0

    for index, (_, row) in enumerate(base_df.iterrows()):
        s0_pred = s0_predictions[index]
        s1_pred = s1_predictions[index]
        b_pred = b_predictions[index]
        c_pred = c_predictions[index]

        p_no_ulcer = float(s0_probabilities[index][s0_index["no_ulcer"]])
        p_ulcer_present = float(s0_probabilities[index][s0_index["ulcer_present"]])
        p_central = float(s1_probabilities[index][s1_index["central_ulcer"]])
        p_noncentral = float(s1_probabilities[index][s1_index["noncentral_ulcer"]])
        p_leq25 = float(b_probabilities[index][b_index["ulcer_leq_25pct"]])
        p_gt25 = float(b_probabilities[index][b_index["greater_than_25pct"]])
        p_lt75 = float(c_probabilities[index][c_index["less_than_75pct"]])
        p_geq75 = float(c_probabilities[index][c_index["ulcer_geq_75pct"]])

        final_probabilities[index, CLASS_TO_INDEX["no_ulcer"]] = p_no_ulcer
        final_probabilities[index, CLASS_TO_INDEX["central_ulcer"]] = p_ulcer_present * p_central
        final_probabilities[index, CLASS_TO_INDEX["ulcer_leq_25pct"]] = p_ulcer_present * p_noncentral * p_leq25 * p_lt75
        final_probabilities[index, CLASS_TO_INDEX["ulcer_leq_50pct"]] = p_ulcer_present * p_noncentral * p_gt25 * p_lt75
        final_probabilities[index, CLASS_TO_INDEX["ulcer_geq_75pct"]] = p_ulcer_present * p_noncentral * p_geq75

        if s0_pred == "no_ulcer":
            final_label = "no_ulcer"
            route = "s0:no_ulcer"
        elif s1_pred == "central_ulcer":
            final_label = "central_ulcer"
            route = "s0:ulcer_present>s1:central_ulcer"
        else:
            if b_pred == "ulcer_leq_25pct" and c_pred == "ulcer_geq_75pct":
                conflicts += 1
                final_label = "ulcer_geq_75pct"
                route = "s0:ulcer_present>s1:noncentral_ulcer>s2ordinal:conflict_geq75_priority"
            elif c_pred == "ulcer_geq_75pct":
                final_label = "ulcer_geq_75pct"
                route = "s0:ulcer_present>s1:noncentral_ulcer>s2ordinal:geq75"
            elif b_pred == "ulcer_leq_25pct":
                final_label = "ulcer_leq_25pct"
                route = "s0:ulcer_present>s1:noncentral_ulcer>s2ordinal:leq25"
            else:
                final_label = "ulcer_leq_50pct"
                route = "s0:ulcer_present>s1:noncentral_ulcer>s2ordinal:leq50"
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
                "s2_leq25_pred": b_pred,
                "s2_geq75_pred": c_pred,
                "correct": bool(row["severity_label"] == final_label),
                "confidence": float(np.max(final_probabilities[index])),
                "raw_image_path": str(row["raw_image_path"]),
            }
        )
    return pd.DataFrame(rows), compute_full_metrics(base_df["severity_label"].tolist(), final_predictions, final_probabilities), {"conflict_count": int(conflicts)}


def mild_bin_diagnostics(predictions: pd.DataFrame) -> dict[str, object]:
    mild = predictions[predictions["true_label"].isin(("ulcer_leq_25pct", "ulcer_leq_50pct"))]
    return {
        "ulcer_leq_25pct_pred_breakdown": mild[mild["true_label"] == "ulcer_leq_25pct"]["pred_label"].value_counts().to_dict(),
        "ulcer_leq_50pct_pred_breakdown": mild[mild["true_label"] == "ulcer_leq_50pct"]["pred_label"].value_counts().to_dict(),
        "no_ulcer_false_negatives": int(((predictions["true_label"] == "no_ulcer") & (predictions["pred_label"] != "no_ulcer")).sum()),
        "no_ulcer_false_positives": int(((predictions["true_label"] != "no_ulcer") & (predictions["pred_label"] == "no_ulcer")).sum()),
    }


def comparison_row(experiment_name: str, metrics: dict[str, object]) -> dict[str, object]:
    return {
        "experiment_name": experiment_name,
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "central_ulcer_recall": float(metrics["central_ulcer_recall"]),
        "no_ulcer_precision": float(metrics["no_ulcer_precision"]),
        "ulcer_leq_25pct_recall": float(metrics["classification_report"]["ulcer_leq_25pct"]["recall"]),
        "ulcer_leq_25pct_f1": float(metrics["classification_report"]["ulcer_leq_25pct"]["f1-score"]),
        "adjacent_class_error_rate": float(metrics["adjacent_class_error_rate"]),
        "delta_vs_hgb_fallback_ba": float(metrics["balanced_accuracy"]) - HGB_FALLBACK["balanced_accuracy"],
        "delta_vs_hgb_fallback_macro_f1": float(metrics["macro_f1"]) - HGB_FALLBACK["macro_f1"],
        "delta_vs_sev_s1_best_ba": float(metrics["balanced_accuracy"]) - BEST_SEV_S1["balanced_accuracy"],
        "delta_vs_sev_s1_best_macro_f1": float(metrics["macro_f1"]) - BEST_SEV_S1["macro_f1"],
        "delta_vs_sev_s2_best_ba": float(metrics["balanced_accuracy"]) - BEST_SEV_S2["balanced_accuracy"],
        "delta_vs_sev_s2_best_macro_f1": float(metrics["macro_f1"]) - BEST_SEV_S2["macro_f1"],
        "delta_vs_strict_reference_ba": float(metrics["balanced_accuracy"]) - STRICT_REFERENCE["balanced_accuracy"],
        "delta_vs_strict_reference_macro_f1": float(metrics["macro_f1"]) - STRICT_REFERENCE["macro_f1"],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    geom_table = pd.read_csv(args.geom_table)
    geom_table["image_id"] = geom_table["image_id"].astype(str)

    flat_stage_metrics = {
        "s0": json.loads((output_root / "metrics" / args.s0_experiment / "test_metrics.json").read_text(encoding="utf-8")),
        "s1": json.loads((output_root / "metrics" / args.s1_experiment / "test_metrics.json").read_text(encoding="utf-8")),
        "s2_flat": json.loads((output_root / "metrics" / args.s2_flat_experiment / "test_metrics.json").read_text(encoding="utf-8")),
    }
    ordinal_stage_metrics = {
        "s0": flat_stage_metrics["s0"],
        "s1": flat_stage_metrics["s1"],
        "s2_leq25": json.loads((output_root / "metrics" / args.s2_leq25_experiment / "test_metrics.json").read_text(encoding="utf-8")),
        "s2_geq75": json.loads((output_root / "metrics" / args.s2_geq75_experiment / "test_metrics.json").read_text(encoding="utf-8")),
    }

    flat_results = {}
    ordinal_results = {}
    ordinal_conflict_summary = {}
    for split_name in ("val", "test"):
        base_df = geom_table[geom_table["split"] == split_name].reset_index(drop=True)
        flat_rows, flat_metrics = combine_flat(base_df, output_root, args.s0_experiment, args.s1_experiment, args.s2_flat_experiment)
        save_outputs(output_root, args.flat_experiment_name, split_name, flat_rows, flat_metrics)
        flat_results[split_name] = (flat_rows, flat_metrics)

        ordinal_rows, ordinal_metrics, ordinal_diag = combine_ordinal(
            base_df,
            output_root,
            args.s0_experiment,
            args.s1_experiment,
            args.s2_leq25_experiment,
            args.s2_geq75_experiment,
        )
        save_outputs(output_root, args.ordinal_experiment_name, split_name, ordinal_rows, ordinal_metrics)
        ordinal_results[split_name] = (ordinal_rows, ordinal_metrics)
        if split_name == "test":
            ordinal_conflict_summary = ordinal_diag

    flat_test_rows, flat_test_metrics = flat_results["test"]
    ordinal_test_rows, ordinal_test_metrics = ordinal_results["test"]
    write_json(output_root / "reports" / args.flat_experiment_name / "stage_test_metrics.json", flat_stage_metrics)
    write_json(output_root / "reports" / args.ordinal_experiment_name / "stage_test_metrics.json", ordinal_stage_metrics)
    write_json(output_root / "debug" / "severity_posthoc" / args.flat_experiment_name / "test_error_diagnostics.json", mild_bin_diagnostics(flat_test_rows))
    ordinal_diagnostics = mild_bin_diagnostics(ordinal_test_rows)
    ordinal_diagnostics.update(ordinal_conflict_summary)
    write_json(output_root / "debug" / "severity_posthoc" / args.ordinal_experiment_name / "test_error_diagnostics.json", ordinal_diagnostics)

    comparison_rows = [
        comparison_row(args.flat_experiment_name, flat_test_metrics),
        comparison_row(args.ordinal_experiment_name, ordinal_test_metrics),
    ]
    comparison_rows.sort(key=lambda row: (row["balanced_accuracy"], row["macro_f1"]), reverse=True)
    debug_dir = output_root / "debug" / "severity_posthoc"
    write_csv_rows(debug_dir / "sev_s3_comparison_summary.csv", comparison_rows)

    lines = [
        "# SEV-S3 S2 Ordinal Rescue",
        "",
        f"- Strict reference: `{STRICT_REFERENCE['name']}` BA {STRICT_REFERENCE['balanced_accuracy']:.4f}, macro F1 {STRICT_REFERENCE['macro_f1']:.4f}",
        f"- SEV-S1 best: `{BEST_SEV_S1['name']}` BA {BEST_SEV_S1['balanced_accuracy']:.4f}, macro F1 {BEST_SEV_S1['macro_f1']:.4f}",
        f"- SEV-S2 best: `{BEST_SEV_S2['name']}` BA {BEST_SEV_S2['balanced_accuracy']:.4f}, macro F1 {BEST_SEV_S2['macro_f1']:.4f}",
        f"- Fallback: `{HGB_FALLBACK['name']}` BA {HGB_FALLBACK['balanced_accuracy']:.4f}, macro F1 {HGB_FALLBACK['macro_f1']:.4f}",
        "",
        "## Combined Comparison",
        "",
        "| Experiment | BA | Macro F1 | No-Ulcer Precision | Central Recall | leq25 Recall | leq25 F1 | Adjacent Error | dBA vs SEV-S2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            "| {experiment_name} | {balanced_accuracy:.4f} | {macro_f1:.4f} | {no_ulcer_precision:.4f} | {central_ulcer_recall:.4f} | {ulcer_leq_25pct_recall:.4f} | {ulcer_leq_25pct_f1:.4f} | {adjacent_class_error_rate:.4f} | {delta_vs_sev_s2_best_ba:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## S2 Stage Metrics",
            "",
            f"- Flat S2 `{args.s2_flat_experiment}`: BA `{flat_stage_metrics['s2_flat']['balanced_accuracy']:.4f}`, macro F1 `{flat_stage_metrics['s2_flat']['macro_f1']:.4f}`",
            f"- Threshold B `{args.s2_leq25_experiment}`: BA `{ordinal_stage_metrics['s2_leq25']['balanced_accuracy']:.4f}`, macro F1 `{ordinal_stage_metrics['s2_leq25']['macro_f1']:.4f}`",
            f"- Threshold C `{args.s2_geq75_experiment}`: BA `{ordinal_stage_metrics['s2_geq75']['balanced_accuracy']:.4f}`, macro F1 `{ordinal_stage_metrics['s2_geq75']['macro_f1']:.4f}`",
            "",
            "## Ordinal Conflict Handling",
            "",
            f"- Conflict count (`<=25` and `>=75` both predicted): `{ordinal_conflict_summary.get('conflict_count', 0)}`",
            "- Resolution rule: `>=75` wins",
            "",
        ]
    )
    for experiment_name, metrics in (
        (args.flat_experiment_name, flat_test_metrics),
        (args.ordinal_experiment_name, ordinal_test_metrics),
    ):
        report = metrics["classification_report"]
        lines.extend(
            [
                f"### `{experiment_name}`",
                "",
                f"- balanced accuracy `{metrics['balanced_accuracy']:.4f}`",
                f"- macro F1 `{metrics['macro_f1']:.4f}`",
                f"- no-ulcer precision `{metrics['no_ulcer_precision']:.4f}`",
                f"- central-ulcer recall `{metrics['central_ulcer_recall']:.4f}`",
                f"- ulcer_leq_25pct recall / F1 `{report['ulcer_leq_25pct']['recall']:.4f} / {report['ulcer_leq_25pct']['f1-score']:.4f}`",
                f"- adjacent-class error rate `{metrics['adjacent_class_error_rate']:.4f}`",
                f"- confusion matrix `{metrics['confusion_matrix']}`",
                "",
            ]
        )
    content = "\n".join(lines)
    write_text(output_root / "reports" / args.ordinal_experiment_name / "test_summary.md", content)
    if args.report_path:
        write_text(args.report_path, content)
    print(json.dumps({"ordinal_experiment_name": args.ordinal_experiment_name, "test_balanced_accuracy": ordinal_test_metrics["balanced_accuracy"], "test_macro_f1": ordinal_test_metrics["macro_f1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
