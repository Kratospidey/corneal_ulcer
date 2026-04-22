from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

from utils_io import write_csv_rows, write_text


STRICT_REFERENCE = {
    "name": "severity5_pattern3__convnextv2_tiny__cornea_crop_scale_v1__severity_first_structured3head_tempered_v1__holdout_v1__seed42",
    "balanced_accuracy": 0.6108918987988756,
    "macro_f1": 0.55424317617866,
}
HGB_FALLBACK = {
    "name": "hgb_fallback",
    "balanced_accuracy": 0.3279649033137405,
    "macro_f1": 0.3327291324180098,
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Aggregate SEV-S1 post-hoc experiment metrics.")
    parser.add_argument("--experiment-name", action="append", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--report-path")
    return parser


def load_metrics(output_root: Path, experiment_name: str) -> dict[str, object]:
    metrics_path = output_root / "metrics" / experiment_name / "test_metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def per_class_lines(metrics: dict[str, object]) -> list[str]:
    report = metrics["classification_report"]
    rows = []
    for class_name in ("no_ulcer", "ulcer_leq_25pct", "ulcer_leq_50pct", "ulcer_geq_75pct", "central_ulcer"):
        rows.append(
            "- `{}`: `{:.4f} / {:.4f}`".format(
                class_name,
                float(report[class_name]["recall"]),
                float(report[class_name]["f1-score"]),
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    rows = []
    for experiment_name in args.experiment_name:
        metrics = load_metrics(output_root, experiment_name)
        rows.append(
            {
                "experiment_name": experiment_name,
                "metrics": metrics,
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "central_ulcer_recall": float(metrics["central_ulcer_recall"]),
                "no_ulcer_precision": float(metrics["no_ulcer_precision"]),
                "adjacent_class_error_rate": float(metrics["adjacent_class_error_rate"]),
                "delta_vs_hgb_fallback_ba": float(metrics["balanced_accuracy"]) - HGB_FALLBACK["balanced_accuracy"],
                "delta_vs_hgb_fallback_macro_f1": float(metrics["macro_f1"]) - HGB_FALLBACK["macro_f1"],
                "delta_vs_strict_reference_ba": float(metrics["balanced_accuracy"]) - STRICT_REFERENCE["balanced_accuracy"],
                "delta_vs_strict_reference_macro_f1": float(metrics["macro_f1"]) - STRICT_REFERENCE["macro_f1"],
            }
        )

    rows.sort(key=lambda row: (row["balanced_accuracy"], row["macro_f1"]), reverse=True)
    debug_dir = output_root / "debug" / "severity_posthoc"
    debug_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(debug_dir / "comparison_summary.csv", rows)

    lines = [
        "# SEV-S1 Post-Hoc Comparison",
        "",
        f"- Strict reference: `{STRICT_REFERENCE['name']}` with BA {STRICT_REFERENCE['balanced_accuracy']:.4f} and macro F1 {STRICT_REFERENCE['macro_f1']:.4f}",
        f"- Fallback reference: `{HGB_FALLBACK['name']}` with BA {HGB_FALLBACK['balanced_accuracy']:.4f} and macro F1 {HGB_FALLBACK['macro_f1']:.4f}",
        "",
        "| Experiment | BA | Macro F1 | Central Recall | No-Ulcer Precision | Adjacent Error | dBA vs fallback | dMacro F1 vs fallback | dBA vs strict | dMacro F1 vs strict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {experiment_name} | {balanced_accuracy:.4f} | {macro_f1:.4f} | {central_ulcer_recall:.4f} | {no_ulcer_precision:.4f} | {adjacent_class_error_rate:.4f} | {delta_vs_hgb_fallback_ba:.4f} | {delta_vs_hgb_fallback_macro_f1:.4f} | {delta_vs_strict_reference_ba:.4f} | {delta_vs_strict_reference_macro_f1:.4f} |".format(
                **row
            )
        )
    lines.extend(["", "## Per-Run Detail", ""])
    for row in rows:
        metrics = row["metrics"]
        lines.extend(
            [
                f"### `{row['experiment_name']}`",
                "",
                f"- balanced accuracy `{row['balanced_accuracy']:.4f}`",
                f"- macro F1 `{row['macro_f1']:.4f}`",
                f"- central-ulcer recall `{row['central_ulcer_recall']:.4f}`",
                f"- no-ulcer precision `{row['no_ulcer_precision']:.4f}`",
                f"- adjacent-class error rate `{row['adjacent_class_error_rate']:.4f}`",
                "- per-class recall / F1",
            ]
        )
        lines.extend(per_class_lines(metrics))
        lines.extend(
            [
                "- confusion matrix",
                "  - labels: `no_ulcer, ulcer_leq_25pct, ulcer_leq_50pct, ulcer_geq_75pct, central_ulcer`",
                f"  - matrix: `{metrics['confusion_matrix']}`",
                "",
            ]
        )
    content = "\n".join(lines)
    write_text(debug_dir / "comparison_summary.md", content)
    if args.report_path:
        write_text(args.report_path, content)
    print(debug_dir / "comparison_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
