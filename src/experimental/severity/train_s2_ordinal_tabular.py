from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from experimental.severity.train_factorized_tabular import (
    compute_metrics,
    feature_columns,
    inverse_frequency_weights,
    midpoint_or_default,
    safe_stat,
    save_split_outputs,
)
from utils_io import write_csv_rows, write_json


STAGE_CLASS_NAMES = {
    "s2_flat": ("ulcer_leq_25pct", "ulcer_leq_50pct", "ulcer_geq_75pct"),
    "s2_leq25": ("ulcer_leq_25pct", "greater_than_25pct"),
    "s2_geq75": ("less_than_75pct", "ulcer_geq_75pct"),
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train SEV-S3 S2 flat and ordinal tabular models.")
    parser.add_argument("--table", required=True)
    parser.add_argument("--model", required=True, choices=("rules", "hgb", "xgb"))
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--selection-metric", default="balanced_accuracy", choices=("balanced_accuracy", "macro_f1"))
    parser.add_argument("--feature-mode", default="all_numeric", choices=("geom_only", "all_numeric"))
    return parser


def infer_stage(table: pd.DataFrame) -> str:
    stage_names = sorted(set(table["stage_name"].dropna().astype(str).tolist()))
    if len(stage_names) != 1:
        raise RuntimeError(f"Expected exactly one stage_name in S2 table, got {stage_names}")
    return stage_names[0]


def extent_score(frame: pd.DataFrame) -> pd.Series:
    return (
        0.28 * frame["response_area_frac_t1_0"].astype(float)
        + 0.18 * frame["response_area_frac_t2_0"].astype(float)
        + 0.10 * frame["response_area_frac_t3_0"].astype(float)
        + 0.16 * frame["response_weighted_area"].astype(float)
        + 0.10 * frame["response_weighted_area_t1_5"].astype(float)
        + 0.08 * frame["lesion_mass_top25_fraction"].astype(float)
        + 0.05 * frame["lesion_mass_top10_fraction"].astype(float)
        + 0.05 * frame["lesion_largest_component_fraction"].astype(float)
    )


def fit_flat_rule_thresholds(train_df: pd.DataFrame) -> dict[str, float]:
    score = extent_score(train_df)
    leq25 = score[train_df["stage_label"] == "ulcer_leq_25pct"]
    leq50 = score[train_df["stage_label"] == "ulcer_leq_50pct"]
    geq75 = score[train_df["stage_label"] == "ulcer_geq_75pct"]
    first = midpoint_or_default(safe_stat(leq25, "median", 0.05), safe_stat(leq50, "median", 0.10), 0.075)
    second = midpoint_or_default(safe_stat(leq50, "median", 0.10), safe_stat(geq75, "median", 0.16), 0.13)
    if first >= second:
        second = first + 1e-4
    return {"extent25_threshold": first, "extent50_threshold": second}


def predict_flat_rules(frame: pd.DataFrame, thresholds: dict[str, float], class_to_index: dict[str, int]) -> tuple[list[str], np.ndarray]:
    predictions: list[str] = []
    probabilities = np.zeros((len(frame), 3), dtype=np.float32)
    score = extent_score(frame)
    for index, value in enumerate(score.tolist()):
        if value <= thresholds["extent25_threshold"]:
            label = "ulcer_leq_25pct"
        elif value <= thresholds["extent50_threshold"]:
            label = "ulcer_leq_50pct"
        else:
            label = "ulcer_geq_75pct"
        predictions.append(label)
        probabilities[index, class_to_index[label]] = 1.0
    return predictions, probabilities


def fit_leq25_rule_thresholds(train_df: pd.DataFrame) -> dict[str, float]:
    score = extent_score(train_df)
    leq25 = score[train_df["stage_label"] == "ulcer_leq_25pct"]
    gt25 = score[train_df["stage_label"] == "greater_than_25pct"]
    return {
        "threshold": midpoint_or_default(
            safe_stat(leq25, "q75", 0.07),
            safe_stat(gt25, "q25", 0.10),
            0.085,
        )
    }


def predict_leq25_rules(frame: pd.DataFrame, thresholds: dict[str, float], class_to_index: dict[str, int]) -> tuple[list[str], np.ndarray]:
    predictions: list[str] = []
    probabilities = np.zeros((len(frame), 2), dtype=np.float32)
    score = extent_score(frame)
    for index, value in enumerate(score.tolist()):
        label = "ulcer_leq_25pct" if value <= thresholds["threshold"] else "greater_than_25pct"
        predictions.append(label)
        probabilities[index, class_to_index[label]] = 1.0
    return predictions, probabilities


def fit_geq75_rule_thresholds(train_df: pd.DataFrame) -> dict[str, float]:
    score = extent_score(train_df)
    lt75 = score[train_df["stage_label"] == "less_than_75pct"]
    geq75 = score[train_df["stage_label"] == "ulcer_geq_75pct"]
    return {
        "threshold": midpoint_or_default(
            safe_stat(lt75, "q75", 0.12),
            safe_stat(geq75, "q25", 0.15),
            0.135,
        )
    }


def predict_geq75_rules(frame: pd.DataFrame, thresholds: dict[str, float], class_to_index: dict[str, int]) -> tuple[list[str], np.ndarray]:
    predictions: list[str] = []
    probabilities = np.zeros((len(frame), 2), dtype=np.float32)
    score = extent_score(frame)
    for index, value in enumerate(score.tolist()):
        label = "ulcer_geq_75pct" if value >= thresholds["threshold"] else "less_than_75pct"
        predictions.append(label)
        probabilities[index, class_to_index[label]] = 1.0
    return predictions, probabilities


def fit_hgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    columns: list[str],
    class_names: tuple[str, ...],
    selection_metric: str,
) -> tuple[HistGradientBoostingClassifier, dict[str, object]]:
    class_to_index = {name: index for index, name in enumerate(class_names)}
    X_train = train_df[columns].to_numpy(dtype=np.float32)
    y_train = train_df["stage_label"].map(class_to_index).to_numpy(dtype=np.int64)
    X_val = val_df[columns].to_numpy(dtype=np.float32)
    y_val = val_df["stage_label"].tolist()
    sample_weight = inverse_frequency_weights(train_df["stage_label"])

    candidates = [
        {"learning_rate": 0.05, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 10, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_depth": 5, "max_iter": 300, "min_samples_leaf": 8, "l2_regularization": 0.1},
        {"learning_rate": 0.1, "max_depth": 3, "max_iter": 150, "min_samples_leaf": 6, "l2_regularization": 0.0},
    ]
    best_model = None
    best_summary = None
    best_primary = float("-inf")
    best_secondary = float("-inf")
    for config in candidates:
        model = HistGradientBoostingClassifier(loss="log_loss", random_state=42, **config)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        probabilities = model.predict_proba(X_val)
        predictions = [class_names[index] for index in np.argmax(probabilities, axis=1)]
        metrics = compute_metrics(class_names, y_val, predictions, probabilities)
        primary = float(metrics[selection_metric])
        secondary = float(metrics["macro_f1" if selection_metric == "balanced_accuracy" else "balanced_accuracy"])
        if primary > best_primary or (math.isclose(primary, best_primary) and secondary > best_secondary):
            best_model = model
            best_summary = {"candidate_config": config, "val_metrics": metrics}
            best_primary = primary
            best_secondary = secondary
    if best_model is None or best_summary is None:
        raise RuntimeError("Failed to fit any HistGradientBoosting candidate.")
    return best_model, best_summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    table = pd.read_csv(args.table)
    stage_name = infer_stage(table)
    class_names = STAGE_CLASS_NAMES[stage_name]
    class_to_index = {name: index for index, name in enumerate(class_names)}
    columns = feature_columns(table, args.feature_mode)

    train_df = table[table["split"] == "train"].reset_index(drop=True)
    val_df = table[table["split"] == "val"].reset_index(drop=True)
    test_df = table[table["split"] == "test"].reset_index(drop=True)
    training_summary: dict[str, object] = {
        "stage_name": stage_name,
        "class_names": list(class_names),
        "feature_mode": args.feature_mode,
        "model": args.model,
        "feature_columns": columns,
        "row_counts": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
    }

    if args.model == "rules":
        if stage_name == "s2_flat":
            thresholds = fit_flat_rule_thresholds(train_df)
            val_predictions, val_probabilities = predict_flat_rules(val_df, thresholds, class_to_index)
            test_predictions, test_probabilities = predict_flat_rules(test_df, thresholds, class_to_index)
        elif stage_name == "s2_leq25":
            thresholds = fit_leq25_rule_thresholds(train_df)
            val_predictions, val_probabilities = predict_leq25_rules(val_df, thresholds, class_to_index)
            test_predictions, test_probabilities = predict_leq25_rules(test_df, thresholds, class_to_index)
        else:
            thresholds = fit_geq75_rule_thresholds(train_df)
            val_predictions, val_probabilities = predict_geq75_rules(val_df, thresholds, class_to_index)
            test_predictions, test_probabilities = predict_geq75_rules(test_df, thresholds, class_to_index)
        training_summary["rule_thresholds"] = thresholds
    elif args.model == "hgb":
        model, selection_summary = fit_hgb(train_df, val_df, columns, class_names, args.selection_metric)
        training_summary["selection_summary"] = selection_summary
        X_val = val_df[columns].to_numpy(dtype=np.float32)
        X_test = test_df[columns].to_numpy(dtype=np.float32)
        val_probabilities = model.predict_proba(X_val)
        test_probabilities = model.predict_proba(X_test)
        val_predictions = [class_names[index] for index in np.argmax(val_probabilities, axis=1)]
        test_predictions = [class_names[index] for index in np.argmax(test_probabilities, axis=1)]
    else:
        raise RuntimeError("xgboost is not supported in SEV-S3 because it is unavailable in the active environment.")

    val_metrics = compute_metrics(class_names, val_df["stage_label"].tolist(), val_predictions, val_probabilities)
    test_metrics = compute_metrics(class_names, test_df["stage_label"].tolist(), test_predictions, test_probabilities)
    training_summary["val_metrics"] = val_metrics
    training_summary["test_metrics"] = test_metrics

    save_split_outputs(output_root, args.experiment_name, "val", class_names, val_df, val_predictions, val_probabilities, val_metrics)
    save_split_outputs(output_root, args.experiment_name, "test", class_names, test_df, test_predictions, test_probabilities, test_metrics)
    write_json(output_root / "reports" / args.experiment_name / "training_summary.json", training_summary)

    debug_dir = output_root / "debug" / "severity_posthoc" / args.experiment_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(debug_dir / "feature_columns.csv", [{"feature_name": column} for column in columns])
    if args.model == "hgb":
        joblib.dump(model, debug_dir / "model.joblib")

    print(json.dumps({"experiment_name": args.experiment_name, "stage_name": stage_name, "test_balanced_accuracy": test_metrics["balanced_accuracy"], "test_macro_f1": test_metrics["macro_f1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
