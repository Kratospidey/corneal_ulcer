from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from experimental.severity.eval_factorized_severity import (
    BEST_SEV_S1,
    CLASS_TO_INDEX,
    FULL_CLASS_NAMES,
    HGB_FALLBACK,
    STRICT_REFERENCE,
    compute_full_metrics,
    save_outputs,
)
from experimental.severity.eval_s2_ordinal_severity import BEST_SEV_S2, stage_predict
from utils_io import write_csv_rows, write_json, write_text


BEST_SEV_S3 = {
    "name": "severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1",
    "balanced_accuracy": 0.4233120368004089,
    "macro_f1": 0.43655327515621634,
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate ultra-narrow SEV-S4 calibration / coupling variants.")
    parser.add_argument("--geom-table", required=True)
    parser.add_argument("--s0-experiment", required=True)
    parser.add_argument("--s1-experiment", required=True)
    parser.add_argument("--s2-leq25-experiment", required=True)
    parser.add_argument("--s2-geq75-experiment", required=True)
    parser.add_argument("--s0cal-experiment-name", required=True)
    parser.add_argument("--s2cal-experiment-name", required=True)
    parser.add_argument("--probcombine-experiment-name", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--report-path")
    return parser


def clipped_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def fit_binary_calibrator(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {
        "fitted": False,
        "class_balance": float(labels.mean()) if labels.size else 0.0,
    }
    if probabilities.size == 0 or len(np.unique(labels)) < 2:
        return result
    features = clipped_logit(probabilities).reshape(-1, 1)
    model = LogisticRegression(random_state=42, class_weight="balanced", solver="lbfgs")
    model.fit(features, labels.astype(np.int64))
    result.update(
        {
            "fitted": True,
            "model": model,
            "coef": float(model.coef_[0][0]),
            "intercept": float(model.intercept_[0]),
        }
    )
    return result


def apply_binary_calibrator(probabilities: np.ndarray, calibrator: dict[str, object]) -> np.ndarray:
    if not calibrator.get("fitted", False):
        return probabilities.astype(np.float64)
    model = calibrator["model"]
    assert isinstance(model, LogisticRegression)
    features = clipped_logit(probabilities).reshape(-1, 1)
    return model.predict_proba(features)[:, 1]


def stage_bundle(
    base_df: pd.DataFrame,
    output_root: Path,
    s0_experiment: str,
    s1_experiment: str,
    s2_leq25_experiment: str,
    s2_geq75_experiment: str,
) -> dict[str, object]:
    s0_predictions, s0_probabilities = stage_predict(base_df, output_root, s0_experiment)
    s1_predictions, s1_probabilities = stage_predict(base_df, output_root, s1_experiment)
    b_predictions, b_probabilities = stage_predict(base_df, output_root, s2_leq25_experiment)
    c_predictions, c_probabilities = stage_predict(base_df, output_root, s2_geq75_experiment)
    return {
        "s0_predictions": s0_predictions,
        "s0_probabilities": s0_probabilities,
        "s1_predictions": s1_predictions,
        "s1_probabilities": s1_probabilities,
        "b_predictions": b_predictions,
        "b_probabilities": b_probabilities,
        "c_predictions": c_predictions,
        "c_probabilities": c_probabilities,
    }


def fit_calibrators(val_df: pd.DataFrame, bundle: dict[str, object]) -> dict[str, dict[str, object]]:
    s0_probabilities = np.asarray(bundle["s0_probabilities"], dtype=np.float64)
    b_probabilities = np.asarray(bundle["b_probabilities"], dtype=np.float64)
    c_probabilities = np.asarray(bundle["c_probabilities"], dtype=np.float64)

    s0_labels = (val_df["severity_label"].to_numpy(dtype=str) == "no_ulcer").astype(np.int64)
    s2_mask = val_df["severity_label"].isin(("ulcer_leq_25pct", "ulcer_leq_50pct", "ulcer_geq_75pct")).to_numpy()
    b_labels = (val_df.loc[s2_mask, "severity_label"].to_numpy(dtype=str) == "ulcer_leq_25pct").astype(np.int64)
    c_labels = (val_df.loc[s2_mask, "severity_label"].to_numpy(dtype=str) == "ulcer_geq_75pct").astype(np.int64)

    return {
        "s0_no_ulcer": fit_binary_calibrator(s0_probabilities[:, 0], s0_labels),
        "s2_leq25": fit_binary_calibrator(b_probabilities[s2_mask, 0], b_labels),
        "s2_geq75": fit_binary_calibrator(c_probabilities[s2_mask, 1], c_labels),
    }


def classification_report_value(metrics: dict[str, object], label: str, field: str) -> float:
    return float(metrics["classification_report"][label][field])


def variant_predictions(
    base_df: pd.DataFrame,
    bundle: dict[str, object],
    calibrators: dict[str, dict[str, object]],
    variant: str,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    s0_predictions = list(bundle["s0_predictions"])
    s1_predictions = list(bundle["s1_predictions"])
    b_predictions = list(bundle["b_predictions"])
    c_predictions = list(bundle["c_predictions"])
    s0_probabilities = np.asarray(bundle["s0_probabilities"], dtype=np.float64)
    s1_probabilities = np.asarray(bundle["s1_probabilities"], dtype=np.float64)
    b_probabilities = np.asarray(bundle["b_probabilities"], dtype=np.float64)
    c_probabilities = np.asarray(bundle["c_probabilities"], dtype=np.float64)

    p_no = s0_probabilities[:, 0]
    if variant in {"s0cal", "probcombine"}:
        p_no = apply_binary_calibrator(p_no, calibrators["s0_no_ulcer"])
    p_ulcer = 1.0 - p_no

    p_central = s1_probabilities[:, 0]
    p_noncentral = s1_probabilities[:, 1]

    p_leq25 = b_probabilities[:, 0]
    if variant in {"s2cal", "probcombine"}:
        p_leq25 = apply_binary_calibrator(p_leq25, calibrators["s2_leq25"])
    p_gt25 = 1.0 - p_leq25

    p_geq75 = c_probabilities[:, 1]
    if variant in {"s2cal", "probcombine"}:
        p_geq75 = apply_binary_calibrator(p_geq75, calibrators["s2_geq75"])
    p_lt75 = 1.0 - p_geq75

    rows: list[dict[str, object]] = []
    final_predictions: list[str] = []
    final_probabilities = np.zeros((len(base_df), len(FULL_CLASS_NAMES)), dtype=np.float32)
    conflict_count = 0

    for index, (_, row) in enumerate(base_df.iterrows()):
        s0_pred_raw = s0_predictions[index]
        s1_pred_raw = s1_predictions[index]
        b_pred_raw = b_predictions[index]
        c_pred_raw = c_predictions[index]

        s0_pred = "no_ulcer" if p_no[index] >= 0.5 else "ulcer_present"
        b_pred = "ulcer_leq_25pct" if p_leq25[index] >= 0.5 else "greater_than_25pct"
        c_pred = "ulcer_geq_75pct" if p_geq75[index] >= 0.5 else "less_than_75pct"

        final_probabilities[index, CLASS_TO_INDEX["no_ulcer"]] = p_no[index]
        final_probabilities[index, CLASS_TO_INDEX["central_ulcer"]] = p_ulcer[index] * p_central[index]
        final_probabilities[index, CLASS_TO_INDEX["ulcer_leq_25pct"]] = p_ulcer[index] * p_noncentral[index] * p_leq25[index] * p_lt75[index]
        final_probabilities[index, CLASS_TO_INDEX["ulcer_leq_50pct"]] = p_ulcer[index] * p_noncentral[index] * p_gt25[index] * p_lt75[index]
        final_probabilities[index, CLASS_TO_INDEX["ulcer_geq_75pct"]] = p_ulcer[index] * p_noncentral[index] * p_geq75[index]

        if variant == "baseline":
            if s0_pred_raw == "no_ulcer":
                final_label = "no_ulcer"
                route = "s0:no_ulcer"
            elif s1_pred_raw == "central_ulcer":
                final_label = "central_ulcer"
                route = "s0:ulcer_present>s1:central_ulcer"
            else:
                if b_pred_raw == "ulcer_leq_25pct" and c_pred_raw == "ulcer_geq_75pct":
                    conflict_count += 1
                    final_label = "ulcer_geq_75pct"
                    route = "s0:ulcer_present>s1:noncentral_ulcer>s2ordinal:conflict_geq75_priority"
                elif c_pred_raw == "ulcer_geq_75pct":
                    final_label = "ulcer_geq_75pct"
                    route = "s0:ulcer_present>s1:noncentral_ulcer>s2ordinal:geq75"
                elif b_pred_raw == "ulcer_leq_25pct":
                    final_label = "ulcer_leq_25pct"
                    route = "s0:ulcer_present>s1:noncentral_ulcer>s2ordinal:leq25"
                else:
                    final_label = "ulcer_leq_50pct"
                    route = "s0:ulcer_present>s1:noncentral_ulcer>s2ordinal:leq50"
        elif variant == "s0cal":
            if s0_pred == "no_ulcer":
                final_label = "no_ulcer"
                route = "s0cal:no_ulcer"
            elif s1_pred_raw == "central_ulcer":
                final_label = "central_ulcer"
                route = "s0cal:ulcer_present>s1:central_ulcer"
            else:
                if b_pred_raw == "ulcer_leq_25pct" and c_pred_raw == "ulcer_geq_75pct":
                    conflict_count += 1
                    final_label = "ulcer_geq_75pct"
                    route = "s0cal:ulcer_present>s1:noncentral_ulcer>s2ordinal:conflict_geq75_priority"
                elif c_pred_raw == "ulcer_geq_75pct":
                    final_label = "ulcer_geq_75pct"
                    route = "s0cal:ulcer_present>s1:noncentral_ulcer>s2ordinal:geq75"
                elif b_pred_raw == "ulcer_leq_25pct":
                    final_label = "ulcer_leq_25pct"
                    route = "s0cal:ulcer_present>s1:noncentral_ulcer>s2ordinal:leq25"
                else:
                    final_label = "ulcer_leq_50pct"
                    route = "s0cal:ulcer_present>s1:noncentral_ulcer>s2ordinal:leq50"
        elif variant == "s2cal":
            if s0_pred_raw == "no_ulcer":
                final_label = "no_ulcer"
                route = "s0:no_ulcer"
            elif s1_pred_raw == "central_ulcer":
                final_label = "central_ulcer"
                route = "s0:ulcer_present>s1:central_ulcer"
            else:
                noncentral_scores = {
                    "ulcer_leq_25pct": p_leq25[index] * p_lt75[index],
                    "ulcer_leq_50pct": p_gt25[index] * p_lt75[index],
                    "ulcer_geq_75pct": p_geq75[index],
                }
                if b_pred == "ulcer_leq_25pct" and c_pred == "ulcer_geq_75pct":
                    conflict_count += 1
                final_label = max(noncentral_scores.items(), key=lambda item: item[1])[0]
                route = f"s0:ulcer_present>s1:noncentral_ulcer>s2cal:{final_label}"
        else:
            full_scores = {
                label: float(final_probabilities[index, CLASS_TO_INDEX[label]])
                for label in FULL_CLASS_NAMES
            }
            final_label = max(full_scores.items(), key=lambda item: item[1])[0]
            route = f"probcombine:argmax:{final_label}"
            if b_pred == "ulcer_leq_25pct" and c_pred == "ulcer_geq_75pct":
                conflict_count += 1

        final_predictions.append(final_label)
        rows.append(
            {
                "image_id": str(row["image_id"]),
                "split": str(row["split"]),
                "true_label": str(row["severity_label"]),
                "pred_label": final_label,
                "route_taken": route,
                "s0_pred_raw": s0_pred_raw,
                "s0_pred_used": s0_pred,
                "s1_pred_raw": s1_pred_raw,
                "s2_leq25_pred_raw": b_pred_raw,
                "s2_geq75_pred_raw": c_pred_raw,
                "s2_leq25_pred_used": b_pred,
                "s2_geq75_pred_used": c_pred,
                "prob_no_ulcer": float(p_no[index]),
                "prob_central_ulcer": float(final_probabilities[index, CLASS_TO_INDEX["central_ulcer"]]),
                "prob_ulcer_leq_25pct": float(final_probabilities[index, CLASS_TO_INDEX["ulcer_leq_25pct"]]),
                "prob_ulcer_leq_50pct": float(final_probabilities[index, CLASS_TO_INDEX["ulcer_leq_50pct"]]),
                "prob_ulcer_geq_75pct": float(final_probabilities[index, CLASS_TO_INDEX["ulcer_geq_75pct"]]),
                "confidence": float(np.max(final_probabilities[index])),
                "correct": bool(row["severity_label"] == final_label),
                "raw_image_path": str(row["raw_image_path"]),
            }
        )

    metrics = compute_full_metrics(base_df["severity_label"].tolist(), final_predictions, final_probabilities)
    diagnostics = {
        "conflict_count": int(conflict_count),
        "ulcer_leq_25pct_pred_breakdown": pd.Series(final_predictions)[base_df["severity_label"] == "ulcer_leq_25pct"].value_counts().to_dict(),
        "ulcer_leq_50pct_pred_breakdown": pd.Series(final_predictions)[base_df["severity_label"] == "ulcer_leq_50pct"].value_counts().to_dict(),
        "no_ulcer_false_negatives": int(((base_df["severity_label"] == "no_ulcer") & (pd.Series(final_predictions) != "no_ulcer")).sum()),
        "no_ulcer_false_positives": int(((base_df["severity_label"] != "no_ulcer") & (pd.Series(final_predictions) == "no_ulcer")).sum()),
    }
    return pd.DataFrame(rows), metrics, diagnostics


def comparison_row(name: str, metrics: dict[str, object]) -> dict[str, object]:
    return {
        "experiment_name": name,
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "no_ulcer_precision": float(metrics["no_ulcer_precision"]),
        "no_ulcer_recall": classification_report_value(metrics, "no_ulcer", "recall"),
        "central_ulcer_recall": float(metrics["central_ulcer_recall"]),
        "ulcer_leq_25pct_recall": classification_report_value(metrics, "ulcer_leq_25pct", "recall"),
        "ulcer_leq_25pct_f1": classification_report_value(metrics, "ulcer_leq_25pct", "f1-score"),
        "adjacent_class_error_rate": float(metrics["adjacent_class_error_rate"]),
        "delta_vs_hgb_fallback_ba": float(metrics["balanced_accuracy"]) - HGB_FALLBACK["balanced_accuracy"],
        "delta_vs_hgb_fallback_macro_f1": float(metrics["macro_f1"]) - HGB_FALLBACK["macro_f1"],
        "delta_vs_sev_s1_best_ba": float(metrics["balanced_accuracy"]) - BEST_SEV_S1["balanced_accuracy"],
        "delta_vs_sev_s1_best_macro_f1": float(metrics["macro_f1"]) - BEST_SEV_S1["macro_f1"],
        "delta_vs_sev_s2_best_ba": float(metrics["balanced_accuracy"]) - BEST_SEV_S2["balanced_accuracy"],
        "delta_vs_sev_s2_best_macro_f1": float(metrics["macro_f1"]) - BEST_SEV_S2["macro_f1"],
        "delta_vs_sev_s3_best_ba": float(metrics["balanced_accuracy"]) - BEST_SEV_S3["balanced_accuracy"],
        "delta_vs_sev_s3_best_macro_f1": float(metrics["macro_f1"]) - BEST_SEV_S3["macro_f1"],
        "delta_vs_strict_reference_ba": float(metrics["balanced_accuracy"]) - STRICT_REFERENCE["balanced_accuracy"],
        "delta_vs_strict_reference_macro_f1": float(metrics["macro_f1"]) - STRICT_REFERENCE["macro_f1"],
    }


def baseline_metrics(output_root: Path) -> dict[str, object]:
    return load_json(output_root / "metrics" / BEST_SEV_S3["name"] / "test_metrics.json")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    geom_table = pd.read_csv(args.geom_table)
    geom_table["image_id"] = geom_table["image_id"].astype(str)

    val_df = geom_table[geom_table["split"] == "val"].reset_index(drop=True)
    test_df = geom_table[geom_table["split"] == "test"].reset_index(drop=True)

    val_bundle = stage_bundle(
        val_df,
        output_root,
        args.s0_experiment,
        args.s1_experiment,
        args.s2_leq25_experiment,
        args.s2_geq75_experiment,
    )
    test_bundle = stage_bundle(
        test_df,
        output_root,
        args.s0_experiment,
        args.s1_experiment,
        args.s2_leq25_experiment,
        args.s2_geq75_experiment,
    )
    calibrators = fit_calibrators(val_df, val_bundle)
    calibrator_report = {
        name: {key: value for key, value in payload.items() if key != "model"}
        for name, payload in calibrators.items()
    }

    experiments = [
        ("s0cal", args.s0cal_experiment_name),
        ("s2cal", args.s2cal_experiment_name),
        ("probcombine", args.probcombine_experiment_name),
    ]
    results: dict[str, tuple[pd.DataFrame, dict[str, object], dict[str, object]]] = {}
    for variant, experiment_name in experiments:
        val_rows, val_metrics, _ = variant_predictions(val_df, val_bundle, calibrators, variant)
        save_outputs(output_root, experiment_name, "val", val_rows, val_metrics)
        test_rows, test_metrics, diagnostics = variant_predictions(test_df, test_bundle, calibrators, variant)
        save_outputs(output_root, experiment_name, "test", test_rows, test_metrics)
        write_json(output_root / "debug" / "severity_posthoc" / experiment_name / "test_error_diagnostics.json", diagnostics)
        write_json(output_root / "reports" / experiment_name / "calibration_summary.json", calibrator_report)
        results[experiment_name] = (test_rows, test_metrics, diagnostics)

    rows = [comparison_row(BEST_SEV_S3["name"], baseline_metrics(output_root))]
    for _, experiment_name in experiments:
        rows.append(comparison_row(experiment_name, results[experiment_name][1]))
    rows.sort(key=lambda row: (row["balanced_accuracy"], row["macro_f1"]), reverse=True)
    write_csv_rows(output_root / "debug" / "severity_posthoc" / "sev_s4_comparison_summary.csv", rows)

    lines = [
        "# SEV-S4 Final Narrow Calibration Pass",
        "",
        f"- Strict reference: `{STRICT_REFERENCE['name']}` BA {STRICT_REFERENCE['balanced_accuracy']:.4f}, macro F1 {STRICT_REFERENCE['macro_f1']:.4f}",
        f"- HGB fallback: `{HGB_FALLBACK['name']}` BA {HGB_FALLBACK['balanced_accuracy']:.4f}, macro F1 {HGB_FALLBACK['macro_f1']:.4f}",
        f"- Best SEV-S1: `{BEST_SEV_S1['name']}` BA {BEST_SEV_S1['balanced_accuracy']:.4f}, macro F1 {BEST_SEV_S1['macro_f1']:.4f}",
        f"- Best SEV-S2: `{BEST_SEV_S2['name']}` BA {BEST_SEV_S2['balanced_accuracy']:.4f}, macro F1 {BEST_SEV_S2['macro_f1']:.4f}",
        f"- Best SEV-S3: `{BEST_SEV_S3['name']}` BA {BEST_SEV_S3['balanced_accuracy']:.4f}, macro F1 {BEST_SEV_S3['macro_f1']:.4f}",
        "",
        "## Calibration Fits",
        "",
    ]
    for name, payload in calibrator_report.items():
        lines.append(
            f"- `{name}`: fitted=`{payload['fitted']}`, class_balance=`{payload['class_balance']:.4f}`"
            + (f", coef=`{payload['coef']:.4f}`, intercept=`{payload['intercept']:.4f}`" if payload["fitted"] else "")
        )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            "| Experiment | BA | Macro F1 | No-Ulcer Precision | No-Ulcer Recall | Central Recall | leq25 Recall | leq25 F1 | Adjacent Error | dBA vs SEV-S3 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {experiment_name} | {balanced_accuracy:.4f} | {macro_f1:.4f} | {no_ulcer_precision:.4f} | {no_ulcer_recall:.4f} | {central_ulcer_recall:.4f} | {ulcer_leq_25pct_recall:.4f} | {ulcer_leq_25pct_f1:.4f} | {adjacent_class_error_rate:.4f} | {delta_vs_sev_s3_best_ba:.4f} |".format(
                **row
            )
        )

    lines.extend(["", "## New Variant Details", ""])
    for _, experiment_name in experiments:
        _, metrics, diagnostics = results[experiment_name]
        lines.extend(
            [
                f"### `{experiment_name}`",
                "",
                f"- balanced accuracy `{metrics['balanced_accuracy']:.4f}`",
                f"- macro F1 `{metrics['macro_f1']:.4f}`",
                f"- no-ulcer precision / recall `{metrics['no_ulcer_precision']:.4f} / {classification_report_value(metrics, 'no_ulcer', 'recall'):.4f}`",
                f"- central-ulcer recall `{metrics['central_ulcer_recall']:.4f}`",
                f"- ulcer_leq_25pct recall / F1 `{classification_report_value(metrics, 'ulcer_leq_25pct', 'recall'):.4f} / {classification_report_value(metrics, 'ulcer_leq_25pct', 'f1-score'):.4f}`",
                f"- adjacent-class error rate `{metrics['adjacent_class_error_rate']:.4f}`",
                f"- conflict count `{diagnostics['conflict_count']}`",
                f"- confusion matrix `{metrics['confusion_matrix']}`",
                "",
            ]
        )

    report_text = "\n".join(lines)
    if args.report_path:
        write_text(Path(args.report_path).resolve(), report_text)
    print(json.dumps({"best_experiment": rows[0]["experiment_name"], "best_balanced_accuracy": rows[0]["balanced_accuracy"], "best_macro_f1": rows[0]["macro_f1"]}))
    return 0


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
