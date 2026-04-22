from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from utils_io import write_csv_rows, write_json, write_text


STAGE_CLASS_NAMES = {
    "s0": ("no_ulcer", "ulcer_present"),
    "s1": ("central_ulcer", "noncentral_ulcer"),
    "s2": ("ulcer_leq_25pct", "ulcer_leq_50pct", "ulcer_geq_75pct"),
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train SEV-S2 factorized post-hoc stage models.")
    parser.add_argument("--table", required=True)
    parser.add_argument("--model", required=True, choices=("rules", "hgb", "xgb"))
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--selection-metric", default="balanced_accuracy", choices=("balanced_accuracy", "macro_f1"))
    parser.add_argument("--feature-mode", default="all_numeric", choices=("geom_only", "all_numeric"))
    return parser


def safe_stat(series: pd.Series, fn_name: str, default: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return float(default)
    if fn_name == "median":
        return float(clean.median())
    if fn_name == "q10":
        return float(clean.quantile(0.10))
    if fn_name == "q25":
        return float(clean.quantile(0.25))
    if fn_name == "q75":
        return float(clean.quantile(0.75))
    if fn_name == "q90":
        return float(clean.quantile(0.90))
    raise ValueError(f"Unsupported stat: {fn_name}")


def midpoint_or_default(left: float, right: float, default: float) -> float:
    if np.isfinite(left) and np.isfinite(right):
        return float((left + right) / 2.0)
    return float(default)


def infer_stage(table: pd.DataFrame) -> str:
    stage_names = sorted(set(table["stage_name"].dropna().astype(str).tolist()))
    if len(stage_names) != 1:
        raise RuntimeError(f"Expected exactly one stage_name in factorized table, got {stage_names}")
    return stage_names[0]


def feature_columns(table: pd.DataFrame, feature_mode: str) -> list[str]:
    excluded = {
        "image_id",
        "split",
        "severity_label",
        "pattern_label",
        "tg_label",
        "raw_image_path",
        "cornea_mask_path",
        "cornea_fit_quality",
        "cornea_fit_source",
        "s0_label",
        "s1_label",
        "s2_label",
        "stage_name",
        "stage_label",
        "factorized_route",
    }
    columns: list[str] = []
    for column in table.columns:
        if column in excluded:
            continue
        if feature_mode == "geom_only" and column.startswith("pattern_"):
            continue
        if feature_mode == "geom_only" and "_x_" in column:
            continue
        if pd.api.types.is_numeric_dtype(table[column]):
            columns.append(column)
    return columns


def inverse_frequency_weights(labels: pd.Series) -> np.ndarray:
    counts = labels.value_counts().to_dict()
    return np.asarray([1.0 / max(1, counts[str(label)]) for label in labels], dtype=np.float32)


def compute_metrics(class_names: tuple[str, ...], y_true: list[str], y_pred: list[str], probabilities: np.ndarray | None) -> dict[str, object]:
    report = classification_report(y_true, y_pred, labels=list(class_names), output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_true, y_pred, labels=list(class_names)).tolist()
    metrics = {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(class_names), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(class_names), average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion,
    }
    if probabilities is not None:
        metrics["mean_confidence"] = float(np.max(probabilities, axis=1).mean())
    return metrics


def prediction_rows(split_df: pd.DataFrame, class_names: tuple[str, ...], y_pred: list[str], probabilities: np.ndarray | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (_, row) in enumerate(split_df.iterrows()):
        entry = {
            "image_id": str(row["image_id"]),
            "split": str(row["split"]),
            "stage_name": str(row["stage_name"]),
            "true_label": str(row["stage_label"]),
            "pred_label": str(y_pred[index]),
            "severity_label": str(row["severity_label"]),
            "factorized_route": str(row["factorized_route"]),
            "correct": bool(row["stage_label"] == y_pred[index]),
        }
        if probabilities is not None:
            entry["confidence"] = float(np.max(probabilities[index]))
            for class_index, class_name in enumerate(class_names):
                entry[f"prob_{class_name}"] = float(probabilities[index][class_index])
        else:
            entry["confidence"] = 1.0
            for class_name in class_names:
                entry[f"prob_{class_name}"] = 1.0 if class_name == y_pred[index] else 0.0
        rows.append(entry)
    return rows


def save_split_outputs(
    output_root: Path,
    experiment_name: str,
    split_name: str,
    class_names: tuple[str, ...],
    split_df: pd.DataFrame,
    y_pred: list[str],
    probabilities: np.ndarray | None,
    metrics: dict[str, object],
) -> None:
    metrics_dir = output_root / "metrics" / experiment_name
    predictions_dir = output_root / "predictions" / experiment_name
    reports_dir = output_root / "reports" / experiment_name
    debug_dir = output_root / "debug" / "severity_posthoc" / experiment_name

    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    write_json(metrics_dir / f"{split_name}_metrics.json", metrics)
    write_csv_rows(predictions_dir / f"{split_name}_predictions.csv", prediction_rows(split_df, class_names, y_pred, probabilities))
    confusion_rows = []
    for true_index, true_label in enumerate(class_names):
        confusion_row = {"true_label": true_label}
        for pred_index, pred_label in enumerate(class_names):
            confusion_row[pred_label] = int(metrics["confusion_matrix"][true_index][pred_index])
        confusion_rows.append(confusion_row)
    write_csv_rows(debug_dir / f"{split_name}_confusion_matrix.csv", confusion_rows)
    write_text(
        reports_dir / f"{split_name}_summary.md",
        "\n".join(
            [
                f"# {experiment_name} {split_name.title()} Summary",
                "",
                f"- Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
                f"- Macro F1: {metrics['macro_f1']:.4f}",
            ]
        ),
    )


def fit_s0_rule_thresholds(train_df: pd.DataFrame) -> dict[str, float]:
    no_ulcer = train_df[train_df["stage_label"] == "no_ulcer"]
    ulcer_present = train_df[train_df["stage_label"] == "ulcer_present"]
    s0_score = 0.55 * train_df["response_area_frac_t0_5"].astype(float) + 0.45 * train_df["response_weighted_area"].astype(float)
    score_no = s0_score[train_df["stage_label"] == "no_ulcer"]
    score_ulcer = s0_score[train_df["stage_label"] == "ulcer_present"]
    return {
        "s0_score_threshold": midpoint_or_default(
            safe_stat(score_no, "q90", 0.0030),
            safe_stat(score_ulcer, "q10", 0.0100),
            0.0060,
        ),
        "peak_z_threshold": midpoint_or_default(
            safe_stat(no_ulcer["response_peak_z"], "q90", 1.5),
            safe_stat(ulcer_present["response_peak_z"], "q10", 2.5),
            2.0,
        ),
    }


def predict_s0_rules(frame: pd.DataFrame, thresholds: dict[str, float], class_to_index: dict[str, int]) -> tuple[list[str], np.ndarray]:
    probabilities = np.zeros((len(frame), 2), dtype=np.float32)
    predictions: list[str] = []
    s0_score = 0.55 * frame["response_area_frac_t0_5"].astype(float) + 0.45 * frame["response_weighted_area"].astype(float)
    for index, (_, row) in enumerate(frame.iterrows()):
        label = "no_ulcer"
        if float(s0_score.iloc[index]) > thresholds["s0_score_threshold"] or float(row["response_peak_z"]) > thresholds["peak_z_threshold"]:
            label = "ulcer_present"
        predictions.append(label)
        probabilities[index, class_to_index[label]] = 1.0
    return predictions, probabilities


def fit_s1_rule_thresholds(train_df: pd.DataFrame) -> dict[str, float]:
    central = train_df[train_df["stage_label"] == "central_ulcer"]
    noncentral = train_df[train_df["stage_label"] == "noncentral_ulcer"]
    central_score = (
        0.45 * train_df["lesion_fraction_in_central"].astype(float)
        + 0.35 * train_df["central_zone_occupancy"].astype(float)
        + 0.20 * train_df["central_response_weighted_area"].astype(float)
    )
    central_values = central_score[train_df["stage_label"] == "central_ulcer"]
    noncentral_values = central_score[train_df["stage_label"] == "noncentral_ulcer"]
    return {
        "central_score_threshold": midpoint_or_default(
            safe_stat(central_values, "median", 0.20),
            safe_stat(noncentral_values, "median", 0.08),
            0.14,
        ),
        "central_distance_threshold": midpoint_or_default(
            safe_stat(central["lesion_min_dist_norm"], "median", 0.15),
            safe_stat(noncentral["lesion_min_dist_norm"], "median", 0.45),
            0.30,
        ),
    }


def predict_s1_rules(frame: pd.DataFrame, thresholds: dict[str, float], class_to_index: dict[str, int]) -> tuple[list[str], np.ndarray]:
    probabilities = np.zeros((len(frame), 2), dtype=np.float32)
    predictions: list[str] = []
    central_score = (
        0.45 * frame["lesion_fraction_in_central"].astype(float)
        + 0.35 * frame["central_zone_occupancy"].astype(float)
        + 0.20 * frame["central_response_weighted_area"].astype(float)
    )
    for index, (_, row) in enumerate(frame.iterrows()):
        label = "noncentral_ulcer"
        if float(central_score.iloc[index]) >= thresholds["central_score_threshold"] and float(row["lesion_min_dist_norm"]) <= thresholds["central_distance_threshold"]:
            label = "central_ulcer"
        predictions.append(label)
        probabilities[index, class_to_index[label]] = 1.0
    return predictions, probabilities


def fit_s2_rule_thresholds(train_df: pd.DataFrame) -> dict[str, float]:
    score = (
        0.50 * train_df["response_area_frac_t0_5"].astype(float)
        + 0.30 * train_df["response_area_frac_t1_0"].astype(float)
        + 0.20 * train_df["response_weighted_area"].astype(float)
    )
    leq25 = score[train_df["stage_label"] == "ulcer_leq_25pct"]
    leq50 = score[train_df["stage_label"] == "ulcer_leq_50pct"]
    geq75 = score[train_df["stage_label"] == "ulcer_geq_75pct"]
    first = midpoint_or_default(safe_stat(leq25, "median", 0.05), safe_stat(leq50, "median", 0.10), 0.075)
    second = midpoint_or_default(safe_stat(leq50, "median", 0.10), safe_stat(geq75, "median", 0.16), 0.13)
    if first >= second:
        second = first + 1e-4
    return {
        "extent25_threshold": first,
        "extent50_threshold": second,
    }


def predict_s2_rules(frame: pd.DataFrame, thresholds: dict[str, float], class_to_index: dict[str, int]) -> tuple[list[str], np.ndarray]:
    probabilities = np.zeros((len(frame), 3), dtype=np.float32)
    predictions: list[str] = []
    score = (
        0.50 * frame["response_area_frac_t0_5"].astype(float)
        + 0.30 * frame["response_area_frac_t1_0"].astype(float)
        + 0.20 * frame["response_weighted_area"].astype(float)
    )
    for index, _ in enumerate(frame.itertuples()):
        value = float(score.iloc[index])
        if value <= thresholds["extent25_threshold"]:
            label = "ulcer_leq_25pct"
        elif value <= thresholds["extent50_threshold"]:
            label = "ulcer_leq_50pct"
        else:
            label = "ulcer_geq_75pct"
        predictions.append(label)
        probabilities[index, class_to_index[label]] = 1.0
    return predictions, probabilities


def fit_stage_hgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    columns: list[str],
    class_names: tuple[str, ...],
    selection_metric: str,
) -> tuple[HistGradientBoostingClassifier, dict[str, object]]:
    class_to_index = {name: index for index, name in enumerate(class_names)}
    candidate_configs = [
        {"learning_rate": 0.05, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 10, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_depth": 5, "max_iter": 300, "min_samples_leaf": 8, "l2_regularization": 0.1},
        {"learning_rate": 0.1, "max_depth": 3, "max_iter": 150, "min_samples_leaf": 6, "l2_regularization": 0.0},
    ]
    X_train = train_df[columns].to_numpy(dtype=np.float32)
    y_train = train_df["stage_label"].map(class_to_index).to_numpy(dtype=np.int64)
    X_val = val_df[columns].to_numpy(dtype=np.float32)
    y_val = val_df["stage_label"].tolist()
    sample_weight = inverse_frequency_weights(train_df["stage_label"])

    best_model: HistGradientBoostingClassifier | None = None
    best_summary: dict[str, object] | None = None
    best_primary = float("-inf")
    best_secondary = float("-inf")
    for config in candidate_configs:
        model = HistGradientBoostingClassifier(loss="log_loss", random_state=42, **config)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        val_probabilities = model.predict_proba(X_val)
        val_predictions = [class_names[index] for index in np.argmax(val_probabilities, axis=1)]
        val_metrics = compute_metrics(class_names, y_val, val_predictions, val_probabilities)
        primary = float(val_metrics[selection_metric])
        secondary = float(val_metrics["macro_f1" if selection_metric == "balanced_accuracy" else "balanced_accuracy"])
        if primary > best_primary or (math.isclose(primary, best_primary) and secondary > best_secondary):
            best_model = model
            best_summary = {"candidate_config": config, "val_metrics": val_metrics}
            best_primary = primary
            best_secondary = secondary
    if best_model is None or best_summary is None:
        raise RuntimeError("Failed to fit any HistGradientBoosting candidate.")
    return best_model, best_summary


def maybe_fit_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    columns: list[str],
    class_names: tuple[str, ...],
    selection_metric: str,
):
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed in the active environment.") from exc

    class_to_index = {name: index for index, name in enumerate(class_names)}
    X_train = train_df[columns].to_numpy(dtype=np.float32)
    y_train = train_df["stage_label"].map(class_to_index).to_numpy(dtype=np.int64)
    X_val = val_df[columns].to_numpy(dtype=np.float32)
    y_val = val_df["stage_label"].tolist()
    sample_weight = inverse_frequency_weights(train_df["stage_label"])
    num_class = len(class_names)
    objective = "binary:logistic" if num_class == 2 else "multi:softprob"
    candidates = [
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 4, "n_estimators": 250, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8},
    ]
    best_model = None
    best_summary = None
    best_primary = float("-inf")
    best_secondary = float("-inf")
    for config in candidates:
        model = XGBClassifier(
            objective=objective,
            num_class=num_class if num_class > 2 else None,
            random_state=42,
            eval_metric="logloss" if num_class == 2 else "mlogloss",
            tree_method="hist",
            **config,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        probabilities = model.predict_proba(X_val)
        if num_class == 2 and probabilities.ndim == 1:
            probabilities = np.stack([1.0 - probabilities, probabilities], axis=1)
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
        raise RuntimeError("Failed to fit any XGBoost candidate.")
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
        if stage_name == "s0":
            thresholds = fit_s0_rule_thresholds(train_df)
            val_predictions, val_probabilities = predict_s0_rules(val_df, thresholds, class_to_index)
            test_predictions, test_probabilities = predict_s0_rules(test_df, thresholds, class_to_index)
        elif stage_name == "s1":
            thresholds = fit_s1_rule_thresholds(train_df)
            val_predictions, val_probabilities = predict_s1_rules(val_df, thresholds, class_to_index)
            test_predictions, test_probabilities = predict_s1_rules(test_df, thresholds, class_to_index)
        else:
            thresholds = fit_s2_rule_thresholds(train_df)
            val_predictions, val_probabilities = predict_s2_rules(val_df, thresholds, class_to_index)
            test_predictions, test_probabilities = predict_s2_rules(test_df, thresholds, class_to_index)
        training_summary["rule_thresholds"] = thresholds
    elif args.model == "hgb":
        model, selection_summary = fit_stage_hgb(train_df, val_df, columns, class_names, args.selection_metric)
        training_summary["selection_summary"] = selection_summary
        X_val = val_df[columns].to_numpy(dtype=np.float32)
        X_test = test_df[columns].to_numpy(dtype=np.float32)
        val_probabilities = model.predict_proba(X_val)
        test_probabilities = model.predict_proba(X_test)
        val_predictions = [class_names[index] for index in np.argmax(val_probabilities, axis=1)]
        test_predictions = [class_names[index] for index in np.argmax(test_probabilities, axis=1)]
    else:
        model, selection_summary = maybe_fit_xgb(train_df, val_df, columns, class_names, args.selection_metric)
        training_summary["selection_summary"] = selection_summary
        X_val = val_df[columns].to_numpy(dtype=np.float32)
        X_test = test_df[columns].to_numpy(dtype=np.float32)
        val_probabilities = model.predict_proba(X_val)
        test_probabilities = model.predict_proba(X_test)
        if len(class_names) == 2 and val_probabilities.ndim == 1:
            val_probabilities = np.stack([1.0 - val_probabilities, val_probabilities], axis=1)
            test_probabilities = np.stack([1.0 - test_probabilities, test_probabilities], axis=1)
        val_predictions = [class_names[index] for index in np.argmax(val_probabilities, axis=1)]
        test_predictions = [class_names[index] for index in np.argmax(test_probabilities, axis=1)]

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
    if args.model in {"hgb", "xgb"}:
        joblib.dump(model, debug_dir / "model.joblib")

    print(json.dumps({"experiment_name": args.experiment_name, "stage_name": stage_name, "test_balanced_accuracy": test_metrics["balanced_accuracy"], "test_macro_f1": test_metrics["macro_f1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
