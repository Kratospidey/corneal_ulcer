from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from utils_io import write_csv_rows, write_json, write_text


S0_LABELS = ("no_ulcer", "ulcer_present")
S1_LABELS = ("central_ulcer", "noncentral_ulcer")
S2_LABELS = ("ulcer_leq_25pct", "ulcer_leq_50pct", "ulcer_geq_75pct")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build SEV-S2 factorized stage tables from a geometry table.")
    parser.add_argument("--table", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def route_labels(table: pd.DataFrame) -> pd.DataFrame:
    routed = table.copy()
    routed["s0_label"] = routed["severity_label"].apply(lambda value: "no_ulcer" if value == "no_ulcer" else "ulcer_present")
    routed["s1_label"] = routed["severity_label"].apply(
        lambda value: "central_ulcer" if value == "central_ulcer" else ("noncentral_ulcer" if value != "no_ulcer" else "")
    )
    routed["s2_label"] = routed["severity_label"].apply(lambda value: value if value in S2_LABELS else "")
    routed["factorized_route"] = routed["severity_label"].apply(
        lambda value: "s0:no_ulcer"
        if value == "no_ulcer"
        else ("s0:ulcer_present>s1:central_ulcer" if value == "central_ulcer" else f"s0:ulcer_present>s1:noncentral_ulcer>s2:{value}")
    )
    return routed


def stage_table(routed: pd.DataFrame, stage_name: str) -> pd.DataFrame:
    if stage_name == "s0":
        frame = routed.copy()
        frame["stage_name"] = "s0"
        frame["stage_label"] = frame["s0_label"]
        return frame
    if stage_name == "s1":
        frame = routed[routed["s0_label"] == "ulcer_present"].copy()
        frame["stage_name"] = "s1"
        frame["stage_label"] = frame["s1_label"]
        return frame.reset_index(drop=True)
    if stage_name == "s2":
        frame = routed[routed["s1_label"] == "noncentral_ulcer"].copy()
        frame["stage_name"] = "s2"
        frame["stage_label"] = frame["s2_label"]
        return frame.reset_index(drop=True)
    raise ValueError(f"Unsupported stage: {stage_name}")


def frame_to_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    return frame.to_dict(orient="records")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    table_path = Path(args.table).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(table_path)
    routed = route_labels(table)
    routed_rows = frame_to_rows(routed)
    write_csv_rows(output_dir / "factorized_master_table.csv", routed_rows)

    summary: dict[str, object] = {
        "source_table": str(table_path),
        "rows": int(len(routed)),
        "severity_counts": {str(key): int(value) for key, value in routed["severity_label"].value_counts().sort_index().items()},
        "s0_counts": {str(key): int(value) for key, value in routed["s0_label"].value_counts().sort_index().items()},
        "s1_counts": {
            str(key): int(value)
            for key, value in routed.loc[routed["s1_label"] != "", "s1_label"].value_counts().sort_index().items()
        },
        "s2_counts": {
            str(key): int(value)
            for key, value in routed.loc[routed["s2_label"] != "", "s2_label"].value_counts().sort_index().items()
        },
        "split_counts": {
            split_name: {
                "s0": int((routed["split"] == split_name).sum()),
                "s1": int(((routed["split"] == split_name) & (routed["s1_label"] != "")).sum()),
                "s2": int(((routed["split"] == split_name) & (routed["s2_label"] != "")).sum()),
            }
            for split_name in ("train", "val", "test")
        },
    }

    report_lines = [
        "# SEV-S2 Factorized Routing",
        "",
        f"- Source table: `{table_path}`",
        "- Routing logic:",
        "  - S0: `no_ulcer` vs `ulcer_present`",
        "  - S1: among `ulcer_present`, `central_ulcer` vs `noncentral_ulcer`",
        "  - S2: among `noncentral_ulcer`, `ulcer_leq_25pct` vs `ulcer_leq_50pct` vs `ulcer_geq_75pct`",
        "",
        "## Counts",
        "",
        f"- S0: `{summary['s0_counts']}`",
        f"- S1: `{summary['s1_counts']}`",
        f"- S2: `{summary['s2_counts']}`",
    ]
    write_json(output_dir / "routing_summary.json", summary)
    write_text(output_dir / "routing_summary.md", "\n".join(report_lines))

    for stage_name in ("s0", "s1", "s2"):
        frame = stage_table(routed, stage_name)
        write_csv_rows(output_dir / f"{stage_name}_table.csv", frame_to_rows(frame))

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
