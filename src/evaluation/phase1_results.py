from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import json
import re

from config_utils import resolve_config, write_text
from data.label_utils import get_task_definition
from experiment_utils import build_experiment_name, write_csv_rows


PHASE1_TAG_PATTERN = re.compile(r"(A\d+)")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Rank pattern Phase 1 experiment results by validation balanced accuracy.")
    parser.add_argument("--config-glob", default="configs/phase1/train_pattern3_phase1_A*.yaml")
    parser.add_argument("--output-root", default="outputs/phase1_pattern_a0a6")
    parser.add_argument("--report-dir", default="outputs/reports/phase1_pattern_a0a6")
    return parser


def _extract_phase_tag(config_path: str | Path) -> str:
    match = PHASE1_TAG_PATTERN.search(Path(config_path).stem)
    if match:
        return match.group(1)
    return Path(config_path).stem


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_phase1_rows(config_glob: str, output_root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    output_root = Path(output_root)
    for config_path in sorted(Path(".").glob(config_glob)):
        config = resolve_config(config_path)
        task_name = str(resolve_config(config["task_config"])["task_name"])
        task_definition = get_task_definition(task_name)
        experiment_name = build_experiment_name({**config, "task_name": task_definition.task_name})
        metric_dir = output_root / "metrics" / experiment_name
        val_metrics = _load_json(metric_dir / "val_metrics.json")
        test_metrics = _load_json(metric_dir / "test_metrics.json")
        training_summary = _load_json(metric_dir / "training_summary.json")
        rows.append(
            {
                "phase_id": _extract_phase_tag(config_path),
                "config_path": str(config_path),
                "experiment_name": experiment_name,
                "val_balanced_accuracy": val_metrics.get("balanced_accuracy"),
                "val_macro_f1": val_metrics.get("macro_f1"),
                "val_weighted_f1": val_metrics.get("weighted_f1"),
                "val_ece": val_metrics.get("ece"),
                "best_epoch": training_summary.get("best_epoch"),
                "tested": bool(test_metrics),
                "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
                "test_macro_f1": test_metrics.get("macro_f1"),
                "test_weighted_f1": test_metrics.get("weighted_f1"),
                "test_ece": test_metrics.get("ece"),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["val_balanced_accuracy"]) if row["val_balanced_accuracy"] is not None else float("-inf"),
            float(row["val_macro_f1"]) if row["val_macro_f1"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    return rows


def render_phase1_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Pattern Phase 1 Ranked Results",
        "",
        "Ranked by validation balanced accuracy first, then validation macro F1. Test metrics are populated only for selected winners that were explicitly evaluated on test.",
        "",
        "| Rank | Phase | Val BA | Val Macro F1 | Best Epoch | Tested | Test BA | Test Macro F1 |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            f"{rank} | {row['phase_id']} | "
            f"{_fmt(row['val_balanced_accuracy'])} | {_fmt(row['val_macro_f1'])} | "
            f"{row['best_epoch'] if row['best_epoch'] is not None else '-'} | "
            f"{'yes' if row['tested'] else 'no'} | "
            f"{_fmt(row['test_balanced_accuracy'])} | {_fmt(row['test_macro_f1'])} |"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = collect_phase1_rows(args.config_glob, args.output_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(report_dir / "ranked_results.csv", rows)
    write_text(report_dir / "ranked_results.md", render_phase1_markdown(rows))
    print(report_dir / "ranked_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
