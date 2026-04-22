from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from utils_io import write_csv_rows, write_json, write_text


CLASS_NAMES = (
    "no_ulcer",
    "ulcer_leq_25pct",
    "ulcer_leq_50pct",
    "ulcer_geq_75pct",
    "central_ulcer",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train SEV-S1 post-hoc tabular baselines.")
    parser.add_argument("--table", required=True)
    parser.add_argument("--model", required=True, choices=("rules", "hgb", "xgb"))
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--selection-metric", default="balanced_accuracy", choices=("balanced_accuracy", "macro_f1"))
    return parser


def midpoint_or_default(left: float, right: float, default: float) -> float:
    if np.isfinite(left) and np.isfinite(right):
        return float((left + right) / 2.0)
    return float(default)


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


def feature_columns(table: pd.DataFrame) -> list[str]:
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
    }
    columns: list[str] = []
    for column in table.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(table[column]):
            columns.append(column)
    return columns


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


def compute_metrics(y_true: list[str], y_pred: list[str], probabilities: np.ndarray | None) -> dict[str, object]:
    labels = list(CLASS_NAMES)
    y_true_index = [CLASS_TO_INDEX[label] for label in y_true]
    y_pred_index = [CLASS_TO_INDEX[label] for label in y_pred]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    metrics = {
        "accuracy": float(np.mean(np.asarray(y_true_index) == np.asarray(y_pred_index))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion,
        "central_ulcer_recall": float(report["central_ulcer"]["recall"]),
        "no_ulcer_precision": float(report["no_ulcer"]["precision"]),
        "adjacent_class_error_rate": adjacent_error_rate(y_true, y_pred),
    }
    if probabilities is not None:
        metrics["mean_confidence"] = float(np.max(probabilities, axis=1).mean())
    return metrics


def prediction_rows(split_df: pd.DataFrame, y_pred: list[str], probabilities: np.ndarray | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (_, row) in enumerate(split_df.iterrows()):
        entry = {
            "image_id": str(row["image_id"]),
            "split": str(row["split"]),
            "true_label": str(row["severity_label"]),
            "pred_label": str(y_pred[index]),
            "correct": bool(row["severity_label"] == y_pred[index]),
            "raw_image_path": str(row["raw_image_path"]),
        }
        if probabilities is not None:
            entry["confidence"] = float(np.max(probabilities[index]))
            for class_index, class_name in enumerate(CLASS_NAMES):
                entry[f"prob_{class_name}"] = float(probabilities[index][class_index])
        else:
            entry["confidence"] = 1.0
            for class_name in CLASS_NAMES:
                entry[f"prob_{class_name}"] = 1.0 if class_name == y_pred[index] else 0.0
        rows.append(entry)
    return rows


def save_split_outputs(
    output_root: Path,
    experiment_name: str,
    split_name: str,
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
    write_csv_rows(predictions_dir / f"{split_name}_predictions.csv", prediction_rows(split_df, y_pred, probabilities))
    confusion_rows = []
    for true_index, true_label in enumerate(CLASS_NAMES):
        confusion_row = {"true_label": true_label}
        for pred_index, pred_label in enumerate(CLASS_NAMES):
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
                f"- Central-ulcer recall: {metrics['central_ulcer_recall']:.4f}",
                f"- No-ulcer precision: {metrics['no_ulcer_precision']:.4f}",
                f"- Adjacent-class error rate: {metrics['adjacent_class_error_rate']:.4f}",
            ]
        ),
    )


def fit_rule_thresholds(train_df: pd.DataFrame) -> dict[str, float]:
    area = "response_area_frac_t0_5"
    central_fraction = "lesion_fraction_in_central"
    central_distance = "lesion_min_dist_norm"
    central_occupancy = "central_zone_occupancy"

    no_ulcer = train_df[train_df["severity_label"] == "no_ulcer"]
    ulcer = train_df[train_df["severity_label"] != "no_ulcer"]
    leq25 = train_df[train_df["severity_label"] == "ulcer_leq_25pct"]
    leq50 = train_df[train_df["severity_label"] == "ulcer_leq_50pct"]
    geq75 = train_df[train_df["severity_label"] == "ulcer_geq_75pct"]
    central = train_df[train_df["severity_label"] == "central_ulcer"]
    non_central_ulcer = train_df[train_df["severity_label"].isin(("ulcer_leq_25pct", "ulcer_leq_50pct", "ulcer_geq_75pct"))]

    central_score = 0.7 * train_df[central_fraction].astype(float) + 0.3 * train_df[central_occupancy].astype(float)
    central_only_score = central_score[train_df["severity_label"] == "central_ulcer"]
    non_central_score = central_score[train_df["severity_label"].isin(("ulcer_leq_25pct", "ulcer_leq_50pct", "ulcer_geq_75pct"))]

    thresholds = {
        "no_ulcer_area_threshold": midpoint_or_default(
            safe_stat(no_ulcer[area], "q90", 0.0010),
            safe_stat(ulcer[area], "q10", 0.0040),
            0.0025,
        ),
        "extent25_threshold": midpoint_or_default(
            safe_stat(leq25[area], "median", 0.0200),
            safe_stat(leq50[area], "median", 0.0500),
            0.0350,
        ),
        "extent50_threshold": midpoint_or_default(
            safe_stat(leq50[area], "median", 0.0500),
            safe_stat(geq75[area], "median", 0.1000),
            0.0750,
        ),
        "central_score_threshold": midpoint_or_default(
            safe_stat(central_only_score, "median", 0.45),
            safe_stat(non_central_score, "median", 0.20),
            0.325,
        ),
        "central_distance_threshold": midpoint_or_default(
            safe_stat(central[central_distance], "median", 0.30),
            safe_stat(non_central_ulcer[central_distance], "median", 0.55),
            0.425,
        ),
        "central_min_area_threshold": safe_stat(central[area], "q25", 0.0200),
    }
    if thresholds["extent25_threshold"] >= thresholds["extent50_threshold"]:
        thresholds["extent50_threshold"] = thresholds["extent25_threshold"] + 1e-4
    return thresholds


def predict_rules(frame: pd.DataFrame, thresholds: dict[str, float]) -> tuple[list[str], np.ndarray]:
    central_score = 0.7 * frame["lesion_fraction_in_central"].astype(float) + 0.3 * frame["central_zone_occupancy"].astype(float)
    predictions: list[str] = []
    probabilities = np.zeros((len(frame), len(CLASS_NAMES)), dtype=np.float32)
    for index, (_, row) in enumerate(frame.iterrows()):
        area = float(row["response_area_frac_t0_5"])
        if area <= thresholds["no_ulcer_area_threshold"]:
            label = "no_ulcer"
        elif (
            area >= thresholds["central_min_area_threshold"]
            and float(central_score.iloc[index]) >= thresholds["central_score_threshold"]
            and float(row["lesion_min_dist_norm"]) <= thresholds["central_distance_threshold"]
        ):
            label = "central_ulcer"
        elif area <= thresholds["extent25_threshold"]:
            label = "ulcer_leq_25pct"
        elif area <= thresholds["extent50_threshold"]:
            label = "ulcer_leq_50pct"
        else:
            label = "ulcer_geq_75pct"
        predictions.append(label)
        probabilities[index, CLASS_TO_INDEX[label]] = 1.0
    return predictions, probabilities


def inverse_frequency_weights(labels: pd.Series) -> np.ndarray:
    counts = labels.value_counts().to_dict()
    return np.asarray([1.0 / max(1, counts[str(label)]) for label in labels], dtype=np.float32)


def fit_hgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    columns: list[str],
    selection_metric: str,
) -> tuple[HistGradientBoostingClassifier, dict[str, object]]:
    candidate_configs = [
        {"learning_rate": 0.05, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 10, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_depth": 5, "max_iter": 300, "min_samples_leaf": 8, "l2_regularization": 0.1},
        {"learning_rate": 0.1, "max_depth": 3, "max_iter": 150, "min_samples_leaf": 6, "l2_regularization": 0.0},
    ]

    X_train = train_df[columns].to_numpy(dtype=np.float32)
    y_train = train_df["severity_label"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    X_val = val_df[columns].to_numpy(dtype=np.float32)
    y_val = val_df["severity_label"].tolist()
    sample_weight = inverse_frequency_weights(train_df["severity_label"])

    best_model: HistGradientBoostingClassifier | None = None
    best_summary: dict[str, object] | None = None
    best_primary = float("-inf")
    best_secondary = float("-inf")

    for config in candidate_configs:
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            random_state=42,
            **config,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        val_probabilities = model.predict_proba(X_val)
        val_predictions = [CLASS_NAMES[index] for index in np.argmax(val_probabilities, axis=1)]
        val_metrics = compute_metrics(y_val, val_predictions, val_probabilities)
        primary = float(val_metrics[selection_metric])
        secondary = float(val_metrics["macro_f1" if selection_metric == "balanced_accuracy" else "balanced_accuracy"])
        if primary > best_primary or (math.isclose(primary, best_primary) and secondary > best_secondary):
            best_model = model
            best_summary = {
                "candidate_config": config,
                "val_metrics": val_metrics,
            }
            best_primary = primary
            best_secondary = secondary

    if best_model is None or best_summary is None:
        raise RuntimeError("Failed to fit any HistGradientBoosting candidate.")
    return best_model, best_summary


def maybe_fit_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    columns: list[str],
    selection_metric: str,
):
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed in the active environment.") from exc

    X_train = train_df[columns].to_numpy(dtype=np.float32)
    y_train = train_df["severity_label"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    X_val = val_df[columns].to_numpy(dtype=np.float32)
    y_val = val_df["severity_label"].tolist()
    sample_weight = inverse_frequency_weights(train_df["severity_label"])

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
            objective="multi:softprob",
            num_class=len(CLASS_NAMES),
            random_state=42,
            eval_metric="mlogloss",
            tree_method="hist",
            **config,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        probabilities = model.predict_proba(X_val)
        predictions = [CLASS_NAMES[index] for index in np.argmax(probabilities, axis=1)]
        metrics = compute_metrics(y_val, predictions, probabilities)
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
    columns = feature_columns(table)

    train_df = table[table["split"] == "train"].reset_index(drop=True)
    val_df = table[table["split"] == "val"].reset_index(drop=True)
    test_df = table[table["split"] == "test"].reset_index(drop=True)
    training_summary: dict[str, object] = {
        "model": args.model,
        "feature_columns": columns,
        "row_counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }

    if args.model == "rules":
        thresholds = fit_rule_thresholds(train_df)
        training_summary["rule_thresholds"] = thresholds
        val_predictions, val_probabilities = predict_rules(val_df, thresholds)
        test_predictions, test_probabilities = predict_rules(test_df, thresholds)
    elif args.model == "hgb":
        model, selection_summary = fit_hgb(train_df, val_df, columns, args.selection_metric)
        training_summary["selection_summary"] = selection_summary
        X_val = val_df[columns].to_numpy(dtype=np.float32)
        X_test = test_df[columns].to_numpy(dtype=np.float32)
        val_probabilities = model.predict_proba(X_val)
        test_probabilities = model.predict_proba(X_test)
        val_predictions = [CLASS_NAMES[index] for index in np.argmax(val_probabilities, axis=1)]
        test_predictions = [CLASS_NAMES[index] for index in np.argmax(test_probabilities, axis=1)]
    else:
        model, selection_summary = maybe_fit_xgb(train_df, val_df, columns, args.selection_metric)
        training_summary["selection_summary"] = selection_summary
        X_val = val_df[columns].to_numpy(dtype=np.float32)
        X_test = test_df[columns].to_numpy(dtype=np.float32)
        val_probabilities = model.predict_proba(X_val)
        test_probabilities = model.predict_proba(X_test)
        val_predictions = [CLASS_NAMES[index] for index in np.argmax(val_probabilities, axis=1)]
        test_predictions = [CLASS_NAMES[index] for index in np.argmax(test_probabilities, axis=1)]

    val_metrics = compute_metrics(val_df["severity_label"].tolist(), val_predictions, val_probabilities)
    test_metrics = compute_metrics(test_df["severity_label"].tolist(), test_predictions, test_probabilities)
    training_summary["val_metrics"] = val_metrics
    training_summary["test_metrics"] = test_metrics

    save_split_outputs(output_root, args.experiment_name, "val", val_df, val_predictions, val_probabilities, val_metrics)
    save_split_outputs(output_root, args.experiment_name, "test", test_df, test_predictions, test_probabilities, test_metrics)
    write_json(output_root / "reports" / args.experiment_name / "training_summary.json", training_summary)

    debug_dir = output_root / "debug" / "severity_posthoc" / args.experiment_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        debug_dir / "feature_columns.csv",
        [{"feature_name": column} for column in columns],
    )

    print(json.dumps({"experiment_name": args.experiment_name, "test_balanced_accuracy": test_metrics["balanced_accuracy"], "test_macro_f1": test_metrics["macro_f1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
