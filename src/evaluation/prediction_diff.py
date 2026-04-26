from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import csv
import json

from config_utils import write_json, write_text


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Compare two prediction tables row-by-row and summarize the differences.")
    parser.add_argument("--left-csv", required=True)
    parser.add_argument("--right-csv", required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-stem", default="prediction_diff")
    return parser


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_prediction_table(path: str | Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"Prediction table is empty: {path}")
    lookup = {row["image_id"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError(f"Duplicate image_id values found in {path}")
    class_names = [column.removeprefix("prob_") for column in rows[0].keys() if column.startswith("prob_")]
    return class_names, lookup


def build_diff_rows(
    left_lookup: dict[str, dict[str, str]],
    right_lookup: dict[str, dict[str, str]],
    left_name: str,
    right_name: str,
) -> list[dict[str, Any]]:
    image_ids = sorted(set(left_lookup.keys()) & set(right_lookup.keys()), key=lambda value: int(value))
    rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        left_row = left_lookup[image_id]
        right_row = right_lookup[image_id]
        rows.append(
            {
                "image_id": image_id,
                "true_label": left_row["true_label"],
                f"{left_name}_pred": left_row["pred_label"],
                f"{left_name}_conf": float(left_row["confidence"]),
                f"{right_name}_pred": right_row["pred_label"],
                f"{right_name}_conf": float(right_row["confidence"]),
                f"{left_name}_correct": left_row["correct"] == "True",
                f"{right_name}_correct": right_row["correct"] == "True",
                "split": left_row["split"],
                "raw_image_path": left_row.get("raw_image_path", ""),
                "cornea_mask_path": left_row.get("cornea_mask_path", ""),
                "official_only_win": left_row["correct"] == "True" and right_row["correct"] != "True",
                "right_only_win": right_row["correct"] == "True" and left_row["correct"] != "True",
                "prediction_changed": left_row["pred_label"] != right_row["pred_label"],
            }
        )
    return rows


def _per_class_recall(rows: dict[str, dict[str, str]], class_names: list[str], model_name: str) -> dict[str, float]:
    totals = {class_name: 0 for class_name in class_names}
    correct = {class_name: 0 for class_name in class_names}
    for row in rows.values():
        true_label = row["true_label"]
        totals[true_label] += 1
        if row["correct"] == "True":
            correct[true_label] += 1
    return {
        class_name: (correct[class_name] / totals[class_name] if totals[class_name] else 0.0)
        for class_name in class_names
    }


def _true_class_confusion(rows: dict[str, dict[str, str]], true_class: str, class_names: list[str]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in class_names}
    for row in rows.values():
        if row["true_label"] != true_class:
            continue
        counts[row["pred_label"]] += 1
    return counts


def build_summary(
    diff_rows: list[dict[str, Any]],
    left_lookup: dict[str, dict[str, str]],
    right_lookup: dict[str, dict[str, str]],
    class_names: list[str],
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left_only = sum(1 for row in diff_rows if row["official_only_win"])
    right_only = sum(1 for row in diff_rows if row["right_only_win"])
    changed = sum(1 for row in diff_rows if row["prediction_changed"])
    summary = {
        "num_samples": len(diff_rows),
        "prediction_changed_count": changed,
        f"{left_name}_correct_{right_name}_wrong": left_only,
        f"{right_name}_correct_{left_name}_wrong": right_only,
        "per_class_recall": {
            left_name: _per_class_recall(left_lookup, class_names, left_name),
            right_name: _per_class_recall(right_lookup, class_names, right_name),
        },
        "point_flaky_mixed_confusion": {
            left_name: _true_class_confusion(left_lookup, "point_flaky_mixed", class_names),
            right_name: _true_class_confusion(right_lookup, "point_flaky_mixed", class_names),
        },
    }
    return summary


def render_markdown(summary: dict[str, Any], left_name: str, right_name: str) -> str:
    left_recall = summary["per_class_recall"][left_name]
    right_recall = summary["per_class_recall"][right_name]
    left_conf = summary["point_flaky_mixed_confusion"][left_name]
    right_conf = summary["point_flaky_mixed_confusion"][right_name]
    lines = [
        "# Prediction Diff Summary",
        "",
        f"- Samples compared: {summary['num_samples']}",
        f"- Prediction changes: {summary['prediction_changed_count']}",
        f"- {left_name} correct while {right_name} wrong: {summary[f'{left_name}_correct_{right_name}_wrong']}",
        f"- {right_name} correct while {left_name} wrong: {summary[f'{right_name}_correct_{left_name}_wrong']}",
        "",
        "## Per-Class Recall",
        "",
        "| Class | " + left_name + " | " + right_name + " |",
        "| --- | ---: | ---: |",
    ]
    for class_name in left_recall:
        lines.append(f"| {class_name} | {left_recall[class_name]:.4f} | {right_recall[class_name]:.4f} |")
    lines.extend(
        [
            "",
            "## point_flaky_mixed Confusion",
            "",
            "| Predicted Label | " + left_name + " | " + right_name + " |",
            "| --- | ---: | ---: |",
        ]
    )
    for class_name in left_conf:
        lines.append(f"| {class_name} | {left_conf[class_name]} | {right_conf[class_name]} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names, left_lookup = _load_prediction_table(args.left_csv)
    right_class_names, right_lookup = _load_prediction_table(args.right_csv)
    if class_names != right_class_names:
        raise ValueError("Prediction tables do not share the same class order.")
    diff_rows = build_diff_rows(left_lookup, right_lookup, args.left_name, args.right_name)
    summary = build_summary(diff_rows, left_lookup, right_lookup, class_names, args.left_name, args.right_name)
    diff_csv_path = output_dir / f"{args.output_stem}.csv"
    with diff_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diff_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diff_rows)
    write_json(output_dir / f"{args.output_stem}.json", summary)
    write_text(output_dir / f"{args.output_stem}.md", render_markdown(summary, args.left_name, args.right_name))
    print(diff_csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
