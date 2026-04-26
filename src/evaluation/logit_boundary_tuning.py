from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import csv
import json

from config_utils import write_json, write_text
from evaluation.calibration import compute_calibration
from evaluation.metrics import compute_classification_metrics


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Tune validation-time decision biases and temperature on saved logits.")
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bias-min", type=float, default=-1.5)
    parser.add_argument("--bias-max", type=float, default=1.5)
    parser.add_argument("--bias-step", type=float, default=0.05)
    parser.add_argument("--temp-min", type=float, default=0.6)
    parser.add_argument("--temp-max", type=float, default=2.5)
    parser.add_argument("--temp-step", type=float, default=0.05)
    return parser


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _softmax(logits):
    import numpy as np  # type: ignore

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _load_logits(path: str | Path):
    import numpy as np  # type: ignore

    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"Prediction table is empty: {path}")
    class_names = [column.removeprefix("logit_") for column in rows[0].keys() if column.startswith("logit_")]
    logits = np.asarray([[float(row[f"logit_{class_name}"]) for class_name in class_names] for row in rows], dtype=float)
    targets = np.asarray([int(row["target_index"]) for row in rows], dtype=int)
    return rows, class_names, logits, targets


def _nll(probabilities, targets):
    import numpy as np  # type: ignore

    probs = probabilities[np.arange(len(targets)), targets]
    return float((-np.log(np.clip(probs, 1e-12, 1.0))).mean())


def _evaluate_logits(logits, targets, class_names):
    probabilities = _softmax(logits)
    y_pred = probabilities.argmax(axis=1).tolist()
    metrics_payload = compute_classification_metrics(targets.tolist(), y_pred, probabilities, class_names)
    calibration = compute_calibration(probabilities, targets.tolist())
    metrics = dict(metrics_payload["metrics"])
    metrics.update(calibration)
    return metrics, probabilities, y_pred


def _frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 10))
        current += step
    return values


def _best_temperature(val_logits, val_targets, temp_values):
    best_temperature = 1.0
    best_nll = float("inf")
    best_metrics = None
    for temperature in temp_values:
        scaled = val_logits / float(temperature)
        metrics, probabilities, _ = _evaluate_logits(scaled, val_targets, class_names=CLASS_NAMES_PLACEHOLDER)
        nll = _nll(probabilities, val_targets)
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
            best_metrics = metrics
    return best_temperature, best_nll, best_metrics


def _search_biases(val_logits, val_targets, class_names, bias_values):
    import numpy as np  # type: ignore

    best_bias = np.zeros(len(class_names), dtype=float)
    best_metrics = None
    best_score = (float("-inf"), float("-inf"))
    if len(class_names) != 3:
        raise ValueError("Bias search currently expects exactly 3 classes.")
    for bias_1 in bias_values:
        for bias_2 in bias_values:
            bias = np.asarray([0.0, float(bias_1), float(bias_2)], dtype=float)
            metrics, _, _ = _evaluate_logits(val_logits + bias, val_targets, class_names)
            score = (float(metrics["balanced_accuracy"]), float(metrics["macro_f1"]))
            if score > best_score:
                best_score = score
                best_bias = bias
                best_metrics = metrics
    return best_bias, best_metrics


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Logit Boundary Tuning",
        "",
        "Validation selected the decision rule by balanced accuracy first, then macro F1.",
        "",
        f"- Best validation temperature by NLL: {summary['temperature_search']['best_temperature']:.4f}",
        f"- Best validation class bias: {summary['bias_search']['best_bias']}",
        "",
        "| Variant | Val BA | Val Macro F1 | Val ECE | Test BA | Test Macro F1 | Test ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in ("baseline", "temperature_only", "bias_only", "bias_plus_temperature"):
        payload = summary["variants"][variant]
        lines.append(
            "| "
            f"{variant} | "
            f"{payload['val']['balanced_accuracy']:.4f} | {payload['val']['macro_f1']:.4f} | {payload['val']['ece']:.4f} | "
            f"{payload['test']['balanced_accuracy']:.4f} | {payload['test']['macro_f1']:.4f} | {payload['test']['ece']:.4f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    global CLASS_NAMES_PLACEHOLDER

    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, class_names, val_logits, val_targets = _load_logits(args.val_csv)
    test_rows, test_class_names, test_logits, test_targets = _load_logits(args.test_csv)
    if class_names != test_class_names:
        raise ValueError("Validation and test prediction tables do not share the same class order.")
    CLASS_NAMES_PLACEHOLDER = class_names

    temp_values = _frange(args.temp_min, args.temp_max, args.temp_step)
    bias_values = _frange(args.bias_min, args.bias_max, args.bias_step)

    baseline_val_metrics, _, _ = _evaluate_logits(val_logits, val_targets, class_names)
    baseline_test_metrics, _, _ = _evaluate_logits(test_logits, test_targets, class_names)

    best_temperature, best_temperature_nll, _ = _best_temperature(val_logits, val_targets, temp_values)
    temp_val_metrics, _, _ = _evaluate_logits(val_logits / best_temperature, val_targets, class_names)
    temp_test_metrics, _, _ = _evaluate_logits(test_logits / best_temperature, test_targets, class_names)

    best_bias, bias_val_metrics = _search_biases(val_logits, val_targets, class_names, bias_values)
    bias_test_metrics, _, _ = _evaluate_logits(test_logits + best_bias, test_targets, class_names)

    combined_val_metrics, _, _ = _evaluate_logits((val_logits + best_bias) / best_temperature, val_targets, class_names)
    combined_test_metrics, combined_test_probs, combined_test_pred = _evaluate_logits(
        (test_logits + best_bias) / best_temperature,
        test_targets,
        class_names,
    )

    tuned_rows = []
    for row, predicted_index, probs in zip(test_rows, combined_test_pred, combined_test_probs.tolist(), strict=True):
        tuned_rows.append(
            {
                "image_id": row["image_id"],
                "true_label": row["true_label"],
                "baseline_pred": row["pred_label"],
                "tuned_pred": class_names[int(predicted_index)],
                "baseline_conf": float(row["confidence"]),
                "tuned_conf": float(max(probs)),
                "raw_image_path": row.get("raw_image_path", ""),
                "cornea_mask_path": row.get("cornea_mask_path", ""),
            }
        )

    summary = {
        "temperature_search": {
            "best_temperature": best_temperature,
            "best_val_nll": best_temperature_nll,
        },
        "bias_search": {
            "best_bias": {class_name: float(value) for class_name, value in zip(class_names, best_bias, strict=True)},
        },
        "variants": {
            "baseline": {"val": baseline_val_metrics, "test": baseline_test_metrics},
            "temperature_only": {"val": temp_val_metrics, "test": temp_test_metrics},
            "bias_only": {"val": bias_val_metrics, "test": bias_test_metrics},
            "bias_plus_temperature": {"val": combined_val_metrics, "test": combined_test_metrics},
        },
    }
    with (output_dir / "official_tuned_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tuned_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tuned_rows)
    write_json(output_dir / "official_logit_tuning_summary.json", summary)
    write_text(output_dir / "official_logit_tuning_summary.md", _render_markdown(summary))
    print(output_dir / "official_logit_tuning_summary.md")
    return 0


CLASS_NAMES_PLACEHOLDER: list[str] = []


if __name__ == "__main__":
    raise SystemExit(main())
