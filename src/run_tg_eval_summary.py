from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import write_json, write_text


TG_CLASS_NAMES = ("no_ulcer", "micro_punctate", "macro_punctate", "coalescent_macro_punctate", "patch_gt_1mm")
PUNCTATE_CLASS_NAMES = TG_CLASS_NAMES[1:4]
TYPE3_NAME = "coalescent_macro_punctate"
TYPE4_NAME = "patch_gt_1mm"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Write TG-focused rescue metrics from evaluation artifacts.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    return parser


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_focus_metrics(prediction_rows: list[dict[str, str]]) -> dict[str, object]:
    label_to_index = {name: index for index, name in enumerate(TG_CLASS_NAMES)}
    y_true = np.asarray([label_to_index[row["true_label"]] for row in prediction_rows], dtype=np.int64)
    y_pred = np.asarray([label_to_index[row["pred_label"]] for row in prediction_rows], dtype=np.int64)

    punctate_indices = [label_to_index[name] for name in PUNCTATE_CLASS_NAMES]
    punctate_mask = np.isin(y_true, punctate_indices)
    punctate_y_true = y_true[punctate_mask]
    punctate_y_pred = y_pred[punctate_mask]

    report = classification_report(y_true, y_pred, target_names=TG_CLASS_NAMES, output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(len(TG_CLASS_NAMES)))).tolist()

    if punctate_y_true.size > 0:
        punctate_balanced_accuracy = float(
            balanced_accuracy_score(
                punctate_y_true,
                punctate_y_pred,
            )
        )
        punctate_macro_f1 = float(
            f1_score(
                punctate_y_true,
                punctate_y_pred,
                labels=punctate_indices,
                average="macro",
                zero_division=0,
            )
        )
    else:
        punctate_balanced_accuracy = 0.0
        punctate_macro_f1 = 0.0

    return {
        "punctate_family_balanced_accuracy": punctate_balanced_accuracy,
        "punctate_family_macro_f1": punctate_macro_f1,
        "type3_recall": float(report[TYPE3_NAME]["recall"]),
        "type3_f1": float(report[TYPE3_NAME]["f1-score"]),
        "type4_recall_guardrail": float(report[TYPE4_NAME]["recall"]),
        "per_class_recall": {name: float(report[name]["recall"]) for name in TG_CLASS_NAMES},
        "per_class_f1": {name: float(report[name]["f1-score"]) for name in TG_CLASS_NAMES},
        "confusion_matrix": confusion,
    }


def build_summary_markdown(experiment_name: str, split_name: str, base_metrics: dict[str, object], focus_metrics: dict[str, object]) -> str:
    return "\n".join(
        [
            f"# {experiment_name} {split_name.title()} TG Rescue Summary",
            "",
            f"- Balanced accuracy: {base_metrics.get('balanced_accuracy')}",
            f"- Macro F1: {base_metrics.get('macro_f1')}",
            f"- Punctate-family balanced accuracy: {focus_metrics['punctate_family_balanced_accuracy']}",
            f"- Punctate-family macro F1: {focus_metrics['punctate_family_macro_f1']}",
            f"- Type3 recall: {focus_metrics['type3_recall']}",
            f"- Type3 F1: {focus_metrics['type3_f1']}",
            f"- Type4 recall guardrail: {focus_metrics['type4_recall_guardrail']}",
            "",
            "## Per-Class Recall",
            "",
            *[f"- {name}: {value}" for name, value in focus_metrics["per_class_recall"].items()],
            "",
            "## Per-Class F1",
            "",
            *[f"- {name}: {value}" for name, value in focus_metrics["per_class_f1"].items()],
            "",
            "## Confusion Matrix",
            "",
            f"- Labels: {', '.join(TG_CLASS_NAMES)}",
            f"- Matrix: {focus_metrics['confusion_matrix']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    prediction_path = output_root / "predictions" / args.experiment_name / f"{args.split}_predictions.csv"
    metrics_path = output_root / "metrics" / args.experiment_name / f"{args.split}_metrics.json"
    report_root = output_root / "reports" / args.experiment_name
    metrics_output_path = output_root / "metrics" / args.experiment_name / f"{args.split}_tg_focus_metrics.json"
    report_output_path = report_root / f"{args.split}_tg_focus_summary.md"

    prediction_rows = load_prediction_rows(prediction_path)
    base_metrics = load_metrics(metrics_path)
    focus_metrics = compute_focus_metrics(prediction_rows)
    payload = {
        "experiment_name": args.experiment_name,
        "split": args.split,
        "balanced_accuracy": base_metrics.get("balanced_accuracy"),
        "macro_f1": base_metrics.get("macro_f1"),
        **focus_metrics,
    }
    write_json(metrics_output_path, payload)
    write_text(report_output_path, build_summary_markdown(args.experiment_name, args.split, base_metrics, focus_metrics))
    print(report_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
