from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import csv

from config_utils import write_text
from utils_io import safe_open_image
from utils_preprocessing import apply_variant


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Create an error atlas comparing two prediction tables.")
    parser.add_argument("--official-csv", required=True)
    parser.add_argument("--challenger-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--thumbnail-dir", required=True)
    parser.add_argument("--preprocessing-mode", default="cornea_crop_scale_v1")
    parser.add_argument("--max-thumbnails-per-group", type=int, default=12)
    return parser


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_prediction_table(path: str | Path) -> dict[str, dict[str, str]]:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"Prediction table is empty: {path}")
    lookup = {row["image_id"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError(f"Duplicate image_id values found in {path}")
    return lookup


def _group_name(official_row: dict[str, str], challenger_row: dict[str, str]) -> str:
    official_correct = official_row["correct"] == "True"
    challenger_correct = challenger_row["correct"] == "True"
    if official_correct and not challenger_correct:
        return "official_correct_w0035_wrong"
    if challenger_correct and not official_correct:
        return "w0035_correct_official_wrong"
    if not official_correct and not challenger_correct:
        return "both_wrong"
    return "both_correct"


def _make_thumbnail(
    row: dict[str, str],
    output_path: Path,
    preprocessing_mode: str,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = safe_open_image(Path(row["raw_image_path"]))
    cornea_mask_path = row.get("cornea_mask_path", "")
    cornea_mask = safe_open_image(Path(cornea_mask_path)) if cornea_mask_path and cornea_mask_path != "nan" else None
    cropped = apply_variant(image, preprocessing_mode, cornea_mask=cornea_mask)
    thumbnail = cropped.copy()
    thumbnail.thumbnail((192, 192))
    thumbnail.save(output_path)
    return str(output_path)


def _changed_case_type(row: dict[str, Any]) -> str:
    true_label = row["true_label"]
    official_pred = row["official_pred"]
    challenger_pred = row["w0035_pred"]
    if row["group"] == "w0035_correct_official_wrong" and true_label == "point_flaky_mixed":
        return "true_pfm_gained_by_w0035"
    if row["group"] == "official_correct_w0035_wrong" and true_label == "flaky":
        return "true_flaky_lost_by_w0035"
    if row["group"] == "official_correct_w0035_wrong" and true_label == "point_like":
        return "true_point_like_lost_by_w0035"
    if row["group"] == "both_wrong":
        if true_label == "flaky" and challenger_pred in {"point_like", "point_flaky_mixed"}:
            return "true_flaky_still_wrong_under_w0035"
        if true_label == "point_like" and challenger_pred == "point_flaky_mixed":
            return "true_point_like_shifted_to_pfm_by_w0035"
        if true_label == "point_flaky_mixed" and official_pred != true_label and challenger_pred != true_label:
            return "true_pfm_unresolved"
    return "other"


def _build_rows(
    official_lookup: dict[str, dict[str, str]],
    challenger_lookup: dict[str, dict[str, str]],
    preprocessing_mode: str,
    thumbnail_dir: Path,
    max_thumbnails_per_group: int,
) -> list[dict[str, Any]]:
    image_ids = sorted(set(official_lookup.keys()) & set(challenger_lookup.keys()), key=lambda value: int(value))
    rows: list[dict[str, Any]] = []
    per_group_counts: dict[str, int] = {}
    for image_id in image_ids:
        official_row = official_lookup[image_id]
        challenger_row = challenger_lookup[image_id]
        group = _group_name(official_row, challenger_row)
        changed_case_type = _changed_case_type(
            {
                "true_label": official_row["true_label"],
                "official_pred": official_row["pred_label"],
                "w0035_pred": challenger_row["pred_label"],
                "group": group,
            }
        )
        thumbnail_path = ""
        per_group_counts[group] = per_group_counts.get(group, 0) + 1
        if group != "both_correct" and per_group_counts[group] <= max_thumbnails_per_group:
            thumbnail_path = _make_thumbnail(
                official_row,
                thumbnail_dir / group / f"{image_id}.png",
                preprocessing_mode=preprocessing_mode,
            )
        rows.append(
            {
                "image_id": image_id,
                "group": group,
                "changed_case_type": changed_case_type,
                "true_label": official_row["true_label"],
                "official_pred": official_row["pred_label"],
                "official_conf": float(official_row["confidence"]),
                "w0035_pred": challenger_row["pred_label"],
                "w0035_conf": float(challenger_row["confidence"]),
                "raw_image_path": official_row.get("raw_image_path", ""),
                "mask_path": official_row.get("cornea_mask_path", ""),
                "crop_thumbnail_path": thumbnail_path,
                "gradcam_path": "",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _count(rows: list[dict[str, Any]], **conditions: str) -> int:
    total = 0
    for row in rows:
        if all(str(row.get(key)) == value for key, value in conditions.items()):
            total += 1
    return total


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    group_counts = {
        "official_correct_w0035_wrong": _count(rows, group="official_correct_w0035_wrong"),
        "w0035_correct_official_wrong": _count(rows, group="w0035_correct_official_wrong"),
        "both_wrong": _count(rows, group="both_wrong"),
        "both_correct": _count(rows, group="both_correct"),
    }
    lines = [
        "# Error Atlas: Official vs w0035",
        "",
        "## Group Counts",
        "",
        f"- official correct, w0035 wrong: {group_counts['official_correct_w0035_wrong']}",
        f"- w0035 correct, official wrong: {group_counts['w0035_correct_official_wrong']}",
        f"- both wrong: {group_counts['both_wrong']}",
        f"- both correct: {group_counts['both_correct']}",
        "",
        "## Targeted Summaries",
        "",
        f"- true flaky lost by w0035: {_count(rows, changed_case_type='true_flaky_lost_by_w0035')}",
        f"- true point_like lost by w0035: {_count(rows, changed_case_type='true_point_like_lost_by_w0035')}",
        f"- true point_flaky_mixed gained by w0035: {_count(rows, changed_case_type='true_pfm_gained_by_w0035')}",
        "",
        "## Changed Or Wrong Cases",
        "",
        "| image_id | group | case type | true | official | w0035 | official conf | w0035 conf | thumbnail |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    interesting_rows = [row for row in rows if row["group"] != "both_correct"]
    for row in interesting_rows:
        thumb = row["crop_thumbnail_path"] or ""
        lines.append(
            f"| {row['image_id']} | {row['group']} | {row['changed_case_type']} | {row['true_label']} | "
            f"{row['official_pred']} | {row['w0035_pred']} | {row['official_conf']:.4f} | {row['w0035_conf']:.4f} | {thumb} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    official_lookup = _load_prediction_table(args.official_csv)
    challenger_lookup = _load_prediction_table(args.challenger_csv)
    rows = _build_rows(
        official_lookup,
        challenger_lookup,
        preprocessing_mode=args.preprocessing_mode,
        thumbnail_dir=Path(args.thumbnail_dir),
        max_thumbnails_per_group=int(args.max_thumbnails_per_group),
    )
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    _write_csv(output_csv, rows)
    write_text(output_md, _render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
