from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import csv

from config_utils import write_json, write_text
from evaluation.calibration import compute_calibration
from evaluation.metrics import compute_classification_metrics


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Fit a tiny val-trained logit stacker over multiple prediction tables.")
    parser.add_argument("--val-csv", action="append", required=True, help="Repeat once per component as name=path")
    parser.add_argument("--test-csv", action="append", required=True, help="Repeat once per component as name=path")
    parser.add_argument("--output-dir", required=True)
    return parser


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_named_paths(items: list[str]) -> dict[str, Path]:
    payload: dict[str, Path] = {}
    for item in items:
        name, raw_path = item.split("=", 1)
        payload[name] = Path(raw_path)
    return payload


def _entropy(probabilities):
    import numpy as np  # type: ignore

    probs = np.clip(probabilities, 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def _margin(probabilities):
    ordered = sorted((float(value) for value in probabilities), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return ordered[0] - ordered[1]


def _load_feature_table(paths: dict[str, Path]):
    import numpy as np  # type: ignore

    component_rows = {name: _read_rows(path) for name, path in paths.items()}
    first_name = next(iter(component_rows))
    reference_rows = component_rows[first_name]
    class_names = [column.removeprefix("logit_") for column in reference_rows[0].keys() if column.startswith("logit_")]
    feature_rows = []
    targets = []
    image_ids = []
    for row_index, base_row in enumerate(reference_rows):
        image_id = base_row["image_id"]
        features: list[float] = []
        for name, rows in component_rows.items():
            row = rows[row_index]
            if row["image_id"] != image_id:
                raise ValueError("Component prediction rows are misaligned.")
            logits = [float(row[f"logit_{class_name}"]) for class_name in class_names]
            probs = [float(row[f"prob_{class_name}"]) for class_name in class_names]
            features.extend(logits)
            features.extend(probs)
            features.append(float(row["confidence"]))
            features.append(_entropy(probs))
            features.append(_margin(probs))
        feature_rows.append(features)
        targets.append(int(base_row["target_index"]))
        image_ids.append(image_id)
    return np.asarray(feature_rows, dtype=float), np.asarray(targets, dtype=int), class_names, image_ids


def _fit_best_model(X_val, y_val):
    from sklearn.linear_model import LogisticRegression, RidgeClassifier  # type: ignore
    from sklearn.metrics import balanced_accuracy_score, f1_score  # type: ignore
    from sklearn.model_selection import StratifiedKFold  # type: ignore
    from sklearn.pipeline import make_pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    import numpy as np  # type: ignore

    candidates = []
    for c_value in (0.05, 0.1, 0.5, 1.0, 2.0, 5.0):
        candidates.append(
            (
                f"logreg_C{c_value:g}",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=c_value,
                        max_iter=4000,
                        solver="lbfgs",
                    ),
                ),
            )
        )
        candidates.append(
            (
                f"ridge_alpha{1.0 / c_value:g}",
                make_pipeline(
                    StandardScaler(),
                    RidgeClassifier(alpha=1.0 / c_value),
                ),
            )
        )

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    best_name = ""
    best_model = None
    best_score = (float("-inf"), float("-inf"))
    for name, model in candidates:
        fold_scores = []
        for train_index, valid_index in splitter.split(X_val, y_val):
            model.fit(X_val[train_index], y_val[train_index])
            preds = model.predict(X_val[valid_index])
            fold_scores.append(
                (
                    balanced_accuracy_score(y_val[valid_index], preds),
                    f1_score(y_val[valid_index], preds, average="macro", zero_division=0),
                )
            )
        score = tuple(float(np.mean(values)) for values in zip(*fold_scores, strict=True))
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model
    assert best_model is not None
    best_model.fit(X_val, y_val)
    return best_name, best_model, best_score


def _evaluate_model(model, X, y, class_names):
    import numpy as np  # type: ignore

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
    else:
        decision = model.decision_function(X)
        if decision.ndim == 1:
            decision = np.stack([-decision, decision], axis=1)
        shifted = decision - decision.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
    y_pred = probabilities.argmax(axis=1).tolist()
    metrics_payload = compute_classification_metrics(y.tolist(), y_pred, probabilities, class_names)
    calibration = compute_calibration(probabilities, y.tolist())
    metrics = dict(metrics_payload["metrics"])
    metrics.update(calibration)
    return metrics, probabilities, y_pred


def _render_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Logit Stacker",
            "",
            f"- Selected model: {summary['selected_model']}",
            f"- CV mean balanced accuracy: {summary['cv_balanced_accuracy']:.4f}",
            f"- CV mean macro F1: {summary['cv_macro_f1']:.4f}",
            "",
            "| Split | Balanced Accuracy | Macro F1 | Weighted F1 | ECE |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| val | {summary['val']['balanced_accuracy']:.4f} | {summary['val']['macro_f1']:.4f} | {summary['val']['weighted_f1']:.4f} | {summary['val']['ece']:.4f} |",
            f"| test | {summary['test']['balanced_accuracy']:.4f} | {summary['test']['macro_f1']:.4f} | {summary['test']['weighted_f1']:.4f} | {summary['test']['ece']:.4f} |",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_paths = _parse_named_paths(args.val_csv)
    test_paths = _parse_named_paths(args.test_csv)
    if set(val_paths.keys()) != set(test_paths.keys()):
        raise ValueError("Validation and test components must match.")

    X_val, y_val, class_names, _ = _load_feature_table(val_paths)
    X_test, y_test, _, test_image_ids = _load_feature_table(test_paths)
    selected_name, model, cv_score = _fit_best_model(X_val, y_val)
    val_metrics, _, _ = _evaluate_model(model, X_val, y_val, class_names)
    test_metrics, test_probabilities, test_pred = _evaluate_model(model, X_test, y_test, class_names)

    prediction_rows = []
    for image_id, target, pred, probs in zip(test_image_ids, y_test.tolist(), test_pred, test_probabilities.tolist(), strict=True):
        prediction_rows.append(
            {
                "image_id": image_id,
                "target_index": target,
                "predicted_index": pred,
                "true_label": class_names[target],
                "pred_label": class_names[pred],
                "confidence": float(max(probs)),
            }
        )
    with (output_dir / "logit_stacker_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    summary = {
        "selected_model": selected_name,
        "cv_balanced_accuracy": float(cv_score[0]),
        "cv_macro_f1": float(cv_score[1]),
        "val": val_metrics,
        "test": test_metrics,
        "components": sorted(val_paths.keys()),
    }
    write_json(output_dir / "logit_stacker_summary.json", summary)
    write_text(output_dir / "logit_stacker_summary.md", _render_markdown(summary))
    print(output_dir / "logit_stacker_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
