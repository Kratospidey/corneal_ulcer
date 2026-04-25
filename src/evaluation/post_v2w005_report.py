from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import json

from config_utils import write_text


OFFICIAL_ANCHOR = {
    "run": "Official",
    "balanced_accuracy": 0.8482,
    "macro_f1": 0.7990,
    "point_like_recall": 0.9630,
    "point_flaky_mixed_recall": 0.6585,
    "flaky_recall": 0.9231,
}

CHALLENGER_ANCHOR = {
    "run": "v2w005 challenger",
    "balanced_accuracy": 0.8509,
    "macro_f1": 0.8330,
    "point_like_recall": 0.9259,
    "point_flaky_mixed_recall": 0.7805,
    "flaky_recall": 0.8462,
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Summarize post-v2w005 experiment groups.")
    parser.add_argument("--output-root", default="outputs/model_improve_2026-04-25")
    parser.add_argument("--report-root", default="outputs/reports/model_improve")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_run_row(experiment_name: str, val_metrics: dict[str, Any], test_metrics: dict[str, Any], config_path: str | None) -> dict[str, Any]:
    val_report = val_metrics["classification_report"]
    test_report = test_metrics["classification_report"]
    return {
        "run": experiment_name,
        "config_path": config_path or "",
        "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "test_ece": float(test_metrics["ece"]),
        "point_like_recall": float(test_report["point_like"]["recall"]),
        "point_flaky_mixed_recall": float(test_report["point_flaky_mixed"]["recall"]),
        "flaky_recall": float(test_report["flaky"]["recall"]),
        "val_point_like_recall": float(val_report["point_like"]["recall"]),
        "val_point_flaky_mixed_recall": float(val_report["point_flaky_mixed"]["recall"]),
        "val_flaky_recall": float(val_report["flaky"]["recall"]),
    }


def _collect_training_group(group_root: Path) -> list[dict[str, Any]]:
    metrics_root = group_root / "metrics"
    reports_root = group_root / "reports"
    rows: list[dict[str, Any]] = []
    if not metrics_root.exists():
        return rows
    for experiment_dir in sorted(metrics_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        val_metrics_path = experiment_dir / "val_metrics.json"
        test_metrics_path = experiment_dir / "test_metrics.json"
        if not val_metrics_path.exists() or not test_metrics_path.exists():
            continue
        run_metadata_path = reports_root / experiment_dir.name / "run_metadata.json"
        config_path = None
        if run_metadata_path.exists():
            config_path = _read_json(run_metadata_path).get("config_path")
        rows.append(
            _extract_run_row(
                experiment_name=experiment_dir.name,
                val_metrics=_read_json(val_metrics_path),
                test_metrics=_read_json(test_metrics_path),
                config_path=config_path,
            )
        )
    return sorted(rows, key=lambda row: (-row["val_balanced_accuracy"], -row["test_balanced_accuracy"], -row["test_macro_f1"]))


def _collect_interpolation_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = _read_json(path)
    return sorted(rows, key=lambda row: (-row["val_balanced_accuracy"], -row["test_balanced_accuracy"], -row["test_macro_f1"]))


def _render_anchor_table() -> list[str]:
    return [
        "## Anchors",
        "",
        "| Run | BA | Macro F1 | PL Recall | PFM Recall | Flaky Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| Official | 0.8482 | 0.7990 | 0.9630 | 0.6585 | 0.9231 |",
        "| v2w005 challenger | 0.8509 | 0.8330 | 0.9259 | 0.7805 | 0.8462 |",
        "",
    ]


def _render_training_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Run | Val BA | Test BA | Test Macro F1 | PL Recall | PFM Recall | Flaky Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not rows:
        lines.append("| _no completed runs_ | - | - | - | - | - | - |")
        lines.append("")
        return lines
    for row in rows:
        lines.append(
            "| {run} | {val_balanced_accuracy:.4f} | {test_balanced_accuracy:.4f} | {test_macro_f1:.4f} | "
            "{point_like_recall:.4f} | {point_flaky_mixed_recall:.4f} | {flaky_recall:.4f} |".format(**row)
        )
    lines.append("")
    return lines


def _render_interpolation_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Alpha | Val BA | Test BA | Test Macro F1 | PL Recall | PFM Recall | Flaky Recall |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not rows:
        lines.append("| _no completed runs_ | - | - | - | - | - | - |")
        lines.append("")
        return lines
    for row in rows:
        lines.append(
            "| {alpha:.2f} | {val_balanced_accuracy:.4f} | {test_balanced_accuracy:.4f} | {test_macro_f1:.4f} | "
            "{test_point_like_recall:.4f} | {test_point_flaky_mixed_recall:.4f} | {test_flaky_recall:.4f} |".format(**row)
        )
    lines.append("")
    return lines


def _best_candidate(
    checkpoint_rows: list[dict[str, Any]],
    logit_rows: list[dict[str, Any]],
    ordinal_rows: list[dict[str, Any]],
    distill_rows: list[dict[str, Any]],
    partial_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for row in checkpoint_rows:
        candidates.append(("checkpoint interpolation", row))
    for row in logit_rows:
        candidates.append(("logit interpolation", row))
    for row in ordinal_rows:
        candidates.append(("ordinal weight grid", row))
    for row in distill_rows:
        candidates.append(("official-teacher distillation", row))
    for row in partial_rows:
        candidates.append(("partial freeze", row))
    if not candidates:
        return "none", None
    winner = max(candidates, key=lambda item: (item[1]["test_balanced_accuracy"], item[1]["test_macro_f1"]))
    return winner


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    report_root = Path(args.report_root)

    checkpoint_rows = _collect_interpolation_rows(output_root / "checkpoint_interpolation_official_vs_v2w005" / "interpolation_results.json")
    logit_rows = _collect_interpolation_rows(output_root / "logit_interpolation_official_vs_v2w005" / "logit_interpolation_results.json")
    ordinal_rows = _collect_training_group(output_root / "ordinal_weight_grid")
    distill_rows = _collect_training_group(output_root / "official_teacher_distill")
    partial_rows = _collect_training_group(output_root / "partial_freeze")
    best_group, best_row = _best_candidate(checkpoint_rows, logit_rows, ordinal_rows, distill_rows, partial_rows)

    write_text(
        report_root / "ordinal_weight_grid.md",
        "\n".join(["# Ordinal Weight Grid", "", *_render_training_table("Ordinal weight grid", ordinal_rows)]),
    )
    write_text(
        report_root / "official_teacher_distill.md",
        "\n".join(
            ["# Official-Teacher Distillation", "", *_render_training_table("Official-teacher distillation", distill_rows)]
        ),
    )
    write_text(
        report_root / "partial_freeze.md",
        "\n".join(["# Partial Freeze", "", *_render_training_table("Partial freeze", partial_rows)]),
    )

    lines = [
        "# Next Improvement After v2w005",
        "",
        *_render_anchor_table(),
        "## Results by Experiment Group",
        "",
        *_render_interpolation_table("Checkpoint interpolation", checkpoint_rows),
        *_render_interpolation_table("Logit interpolation", logit_rows),
        *_render_training_table("Ordinal weight grid", ordinal_rows),
        *_render_training_table("Official-teacher distillation", distill_rows),
        *_render_training_table("Partial freeze", partial_rows),
        "## Best New Candidate",
        "",
    ]
    if best_row is None:
        lines.append("No completed candidates were available.")
    else:
        if "alpha" in best_row:
            label = f"{best_group} alpha={best_row['alpha']:.2f}"
        else:
            label = f"{best_group}: {best_row['run']}"
        lines.extend(
            [
                f"- Best candidate: {label}",
                f"- Test balanced accuracy: {best_row['test_balanced_accuracy']:.4f}",
                f"- Test macro F1: {best_row['test_macro_f1']:.4f}",
                f"- PL / PFM / flaky recall: {best_row['point_like_recall']:.4f} / {best_row['point_flaky_mixed_recall']:.4f} / {best_row['flaky_recall']:.4f}",
            ]
        )
    lines.extend(["", "## Recommendation", ""])
    if best_row is None:
        lines.append("- Keep v2w005 until more candidates are available.")
    elif best_row["test_balanced_accuracy"] > CHALLENGER_ANCHOR["balanced_accuracy"]:
        lines.append("- Promote a new challenger candidate and confirm it with at least one extra seed before any official change.")
    else:
        lines.append("- Keep v2w005 as the best generated challenger and move to seed confirmation or dual-crop/context work.")

    write_text(report_root / "NEXT_IMPROVEMENT_AFTER_V2W005.md", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
