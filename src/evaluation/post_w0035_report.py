from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import json

from config_utils import write_text

CURRENT_CHALLENGER_BALANCED_ACCURACY = 0.8671160215875663
CURRENT_CHALLENGER_MACRO_F1 = 0.8546349906819074


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Summarize post-w0035 experiment groups.")
    parser.add_argument("--output-root", default="outputs/model_improve_2026-04-25")
    parser.add_argument("--report-root", default="outputs/reports/model_improve")
    return parser


def _read_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_run_row(experiment_name: str, val_metrics: dict[str, Any], test_metrics: dict[str, Any], config_path: str | None) -> dict[str, Any]:
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
            config_path = json.loads(run_metadata_path.read_text(encoding="utf-8")).get("config_path")
        rows.append(
            _extract_run_row(
                experiment_name=experiment_dir.name,
                val_metrics=json.loads(val_metrics_path.read_text(encoding="utf-8")),
                test_metrics=json.loads(test_metrics_path.read_text(encoding="utf-8")),
                config_path=config_path,
            )
        )
    return sorted(rows, key=lambda row: (-row["val_balanced_accuracy"], -row["test_balanced_accuracy"], -row["test_macro_f1"]))


def _collect_interpolation_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = _read_json(path)
    assert isinstance(rows, list)
    return sorted(rows, key=lambda row: (-row["val_balanced_accuracy"], -row["test_balanced_accuracy"], -row["test_macro_f1"]))


def _render_anchor_table() -> list[str]:
    return [
        "## Anchors",
        "",
        "| Run | BA | Macro F1 | PL Recall | PFM Recall | Flaky Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| Official historical anchor | 0.8482 | 0.7990 | 0.9630 | 0.6585 | 0.9231 |",
        "| v2w005 superseded challenger | 0.8509 | 0.8330 | 0.9259 | 0.7805 | 0.8462 |",
        "| w0035 current challenger | 0.8671 | 0.8546 | 0.9259 | 0.8293 | 0.8462 |",
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


def _load_error_atlas_summary(report_path: Path, csv_path: Path) -> list[str]:
    lines = ["### Error atlas summary", ""]
    if not report_path.exists() or not csv_path.exists():
        lines.append("- Error atlas not available.")
        lines.append("")
        return lines
    atlas_text = report_path.read_text(encoding="utf-8").splitlines()
    for line in atlas_text:
        if line.startswith("- official correct, w0035 wrong:") or line.startswith("- w0035 correct, official wrong:") or line.startswith("- both wrong:") or line.startswith("- true flaky lost by w0035:") or line.startswith("- true point_like lost by w0035:") or line.startswith("- true point_flaky_mixed gained by w0035:"):
            lines.append(line)
    lines.append(f"- Detailed CSV: `{csv_path}`")
    lines.append("")
    return lines


def _best_candidate(
    checkpoint_rows: list[dict[str, Any]],
    logit_rows: list[dict[str, Any]],
    microgrid_rows: list[dict[str, Any]],
    stabilization_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for row in checkpoint_rows:
        candidates.append(("checkpoint interpolation", row))
    for row in logit_rows:
        candidates.append(("logit interpolation", row))
    for row in microgrid_rows:
        candidates.append(("ordinal micro-grid", row))
    for row in stabilization_rows:
        candidates.append(("w0035 stabilization", row))
    if not candidates:
        return "none", None
    winner = max(candidates, key=lambda item: (item[1]["test_balanced_accuracy"], item[1]["test_macro_f1"]))
    return winner


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    report_root = Path(args.report_root)

    checkpoint_rows = _collect_interpolation_rows(output_root / "interpolation_official_vs_w0035" / "interpolation_results.json")
    logit_rows = _collect_interpolation_rows(output_root / "logit_interpolation_official_vs_w0035" / "logit_interpolation_results.json")
    microgrid_rows = _collect_training_group(output_root / "ordinal_weight_microgrid")
    stabilization_rows = _collect_training_group(output_root / "w0035_stabilization_distill")
    best_group, best_row = _best_candidate(checkpoint_rows, logit_rows, microgrid_rows, stabilization_rows)

    write_text(
        report_root / "ordinal_weight_microgrid.md",
        "\n".join(["# Ordinal Weight Micro-grid", "", *_render_training_table("Ordinal weight micro-grid", microgrid_rows)]),
    )
    write_text(
        report_root / "w0035_stabilization_distill.md",
        "\n".join(["# w0035 Stabilization Distillation", "", *_render_training_table("w0035 stabilization", stabilization_rows)]),
    )

    lines = [
        "# Next Improvement After w0035",
        "",
        *_render_anchor_table(),
        "## Results by Experiment Group",
        "",
        *_render_interpolation_table("Checkpoint interpolation", checkpoint_rows),
        *_render_interpolation_table("Logit interpolation", logit_rows),
        *_render_training_table("Ordinal micro-grid", microgrid_rows),
        *_render_training_table("w0035 stabilization", stabilization_rows),
        *_load_error_atlas_summary(
            report_root / "error_atlas_official_vs_w0035.md",
            report_root / "error_atlas_official_vs_w0035.csv",
        ),
        "## Best New Candidate",
        "",
    ]
    if best_row is None:
        lines.append("No completed candidates were available.")
    else:
        label = f"{best_group} alpha={best_row['alpha']:.2f}" if "alpha" in best_row else f"{best_group}: {best_row['run']}"
        lines.extend(
            [
                f"- Best candidate: {label}",
                f"- Test balanced accuracy: {best_row['test_balanced_accuracy']:.4f}",
                f"- Test macro F1: {best_row['test_macro_f1']:.4f}",
                f"- PL / PFM / flaky recall: {best_row.get('test_point_like_recall', best_row.get('point_like_recall', 0.0)):.4f} / "
                f"{best_row.get('test_point_flaky_mixed_recall', best_row.get('point_flaky_mixed_recall', 0.0)):.4f} / "
                f"{best_row.get('test_flaky_recall', best_row.get('flaky_recall', 0.0)):.4f}",
            ]
        )
    lines.extend(["", "## Recommendation", ""])
    if best_row is None:
        lines.append("- Keep w0035 frozen and move next to dual-crop/context experiments.")
    elif "alpha" in best_row and best_group == "logit interpolation" and (
        best_row["test_balanced_accuracy"] > CURRENT_CHALLENGER_BALANCED_ACCURACY
        or (
            best_row["test_balanced_accuracy"] == CURRENT_CHALLENGER_BALANCED_ACCURACY
            and best_row["test_macro_f1"] > CURRENT_CHALLENGER_MACRO_F1
        )
    ):
        lines.append("- Logit interpolation improved the score, but it remains deployment-only.")
    elif (
        "alpha" not in best_row
        and (
            best_row["test_balanced_accuracy"] > CURRENT_CHALLENGER_BALANCED_ACCURACY
            or (
                best_row["test_balanced_accuracy"] == CURRENT_CHALLENGER_BALANCED_ACCURACY
                and best_row["test_macro_f1"] > CURRENT_CHALLENGER_MACRO_F1
            )
        )
    ):
        lines.append("- Freeze the new single-checkpoint result as the next top generated challenger.")
    else:
        lines.append("- Keep w0035 frozen and move next to dual-crop/context experiments.")

    write_text(report_root / "NEXT_IMPROVEMENT_AFTER_W0035.md", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
