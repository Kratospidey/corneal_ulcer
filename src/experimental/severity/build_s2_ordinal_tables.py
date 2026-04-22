from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from utils_io import write_csv_rows, write_json, write_text


S2_CLASSES = ("ulcer_leq_25pct", "ulcer_leq_50pct", "ulcer_geq_75pct")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build SEV-S3 S2-specific flat and ordinal training tables.")
    parser.add_argument("--table", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def frame_to_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    return frame.to_dict(orient="records")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    table_path = Path(args.table).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(table_path)
    s2 = table[table["severity_label"].isin(S2_CLASSES)].copy().reset_index(drop=True)

    s2_flat = s2.copy()
    s2_flat["stage_name"] = "s2_flat"
    s2_flat["stage_label"] = s2_flat["severity_label"]
    s2_flat["ordinal_route"] = s2_flat["severity_label"].apply(lambda value: f"flat:{value}")
    s2_flat["factorized_route"] = s2_flat["ordinal_route"]

    s2_leq25 = s2.copy()
    s2_leq25["stage_name"] = "s2_leq25"
    s2_leq25["stage_label"] = s2_leq25["severity_label"].apply(
        lambda value: "ulcer_leq_25pct" if value == "ulcer_leq_25pct" else "greater_than_25pct"
    )
    s2_leq25["ordinal_route"] = s2_leq25["severity_label"].apply(
        lambda value: "leq25" if value == "ulcer_leq_25pct" else "gt25"
    )
    s2_leq25["factorized_route"] = s2_leq25["ordinal_route"]

    s2_geq75 = s2.copy()
    s2_geq75["stage_name"] = "s2_geq75"
    s2_geq75["stage_label"] = s2_geq75["severity_label"].apply(
        lambda value: "ulcer_geq_75pct" if value == "ulcer_geq_75pct" else "less_than_75pct"
    )
    s2_geq75["ordinal_route"] = s2_geq75["severity_label"].apply(
        lambda value: "geq75" if value == "ulcer_geq_75pct" else "lt75"
    )
    s2_geq75["factorized_route"] = s2_geq75["ordinal_route"]

    write_csv_rows(output_dir / "s2_flat.csv", frame_to_rows(s2_flat))
    write_csv_rows(output_dir / "s2_leq25.csv", frame_to_rows(s2_leq25))
    write_csv_rows(output_dir / "s2_geq75.csv", frame_to_rows(s2_geq75))

    summary = {
        "source_table": str(table_path),
        "rows": int(len(s2)),
        "split_counts": {
            split_name: int((s2["split"] == split_name).sum())
            for split_name in ("train", "val", "test")
        },
        "flat_counts": {str(key): int(value) for key, value in s2_flat["stage_label"].value_counts().sort_index().items()},
        "leq25_counts": {str(key): int(value) for key, value in s2_leq25["stage_label"].value_counts().sort_index().items()},
        "geq75_counts": {str(key): int(value) for key, value in s2_geq75["stage_label"].value_counts().sort_index().items()},
    }
    write_json(output_dir / "routing_summary.json", summary)
    write_text(
        output_dir / "routing_summary.md",
        "\n".join(
            [
                "# SEV-S3 S2 Routing",
                "",
                f"- Source table: `{table_path}`",
                "- Flat task: `ulcer_leq_25pct` vs `ulcer_leq_50pct` vs `ulcer_geq_75pct`",
                "- Threshold B: `ulcer_leq_25pct` vs `greater_than_25pct`",
                "- Threshold C: `less_than_75pct` vs `ulcer_geq_75pct`",
                "",
                f"- Flat counts: `{summary['flat_counts']}`",
                f"- Threshold B counts: `{summary['leq25_counts']}`",
                f"- Threshold C counts: `{summary['geq75_counts']}`",
            ]
        ),
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
