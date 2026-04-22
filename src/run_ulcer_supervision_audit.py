from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import csv

from PIL import Image, ImageDraw, ImageOps  # type: ignore

from config_utils import write_text
from experiment_utils import write_csv_rows
from ulcer_supervision_audit import classify_ulcer_supervision_row
from utils_io import safe_open_image
from utils_preprocessing import normalize_ulcer_mask


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Audit ulcer-mask supervision coverage and prepare review artifacts.")
    parser.add_argument("--manifest", default="data/interim/manifests/manifest.csv")
    parser.add_argument("--split-file", default="data/interim/split_files/pattern_3class_holdout.csv")
    parser.add_argument("--output-root", default="outputs/debug/2026-04-19_model_hardening/ulcer_supervision")
    parser.add_argument("--render-limit", type=int, default=48)
    parser.add_argument("--generate-empty-templates", action="store_true")
    return parser


def _load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _split_lookup(split_rows: list[dict[str, str]]) -> dict[str, str]:
    return {str(row["image_id"]): str(row["split"]) for row in split_rows}


def _bool_from_manifest(value: str) -> bool:
    return value.strip().lower() == "true"


def _mask_validity(mask_path: str) -> tuple[bool, float | None]:
    if not mask_path:
        return False, None
    mask = normalize_ulcer_mask(safe_open_image(Path(mask_path)))
    mask_array = ImageOps.grayscale(mask).point(lambda pixel: 1 if pixel > 127 else 0)
    pixels = list(mask_array.getdata())
    if not pixels:
        return False, None
    lesion_fraction = sum(pixels) / len(pixels)
    valid = 0.0 < lesion_fraction < 1.0
    return valid, float(lesion_fraction)


def _build_audit_rows(
    manifest_rows: list[dict[str, str]],
    split_map: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in manifest_rows:
        ulcer_mask_valid, lesion_fraction = _mask_validity(str(row.get("ulcer_mask_path", "")))
        base = {
            "image_id": str(row["image_id"]),
            "split": split_map.get(str(row["image_id"]), "unsplit"),
            "task_pattern_3class": str(row["task_pattern_3class"]),
            "task_binary": str(row["task_binary"]),
            "task_severity_5class": str(row["task_severity_5class"]),
            "raw_image_path": str(row["raw_image_path"]),
            "cornea_mask_path": str(row.get("cornea_mask_path", "")),
            "cornea_overlay_path": str(row.get("cornea_overlay_path", "")),
            "ulcer_mask_path": str(row.get("ulcer_mask_path", "")),
            "ulcer_overlay_path": str(row.get("ulcer_overlay_path", "")),
            "has_cornea_mask": _bool_from_manifest(str(row["has_cornea_mask"])),
            "has_ulcer_mask": _bool_from_manifest(str(row["has_ulcer_mask"])),
            "ulcer_mask_valid": ulcer_mask_valid,
            "ulcer_lesion_fraction": lesion_fraction,
        }
        rows.append(classify_ulcer_supervision_row(base))
    return rows


def _summary_rows(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    counter: Counter[tuple[str, ...]] = Counter(tuple(str(row[key]) for key in keys) for row in rows)
    output = []
    for key_values, count in sorted(counter.items()):
        summary_row = {key: value for key, value in zip(keys, key_values, strict=True)}
        summary_row["count"] = count
        output.append(summary_row)
    return output


def _review_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if bool(row["needs_review"])]


def _make_tile(path: str, label: str, size: tuple[int, int]) -> Image.Image:
    tile = Image.new("RGB", size, (18, 18, 18))
    draw = ImageDraw.Draw(tile)
    if path and Path(path).exists():
        image = safe_open_image(Path(path)).convert("RGB")
        image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
        tile.paste(image)
    draw.rectangle((0, size[1] - 26, size[0], size[1]), fill=(0, 0, 0))
    draw.text((8, size[1] - 20), label, fill=(255, 255, 255))
    return tile


def _render_review_panels(rows: list[dict[str, object]], output_dir: Path, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows[:limit]:
        tile_size = (256, 256)
        canvas = Image.new("RGB", (tile_size[0] * 3, tile_size[1] + 72), (245, 245, 245))
        tiles = [
            _make_tile(str(row["raw_image_path"]), "raw", tile_size),
            _make_tile(str(row["cornea_overlay_path"]), "cornea overlay", tile_size),
            _make_tile(str(row["ulcer_overlay_path"]), "ulcer overlay", tile_size),
        ]
        for index, tile in enumerate(tiles):
            canvas.paste(tile, (index * tile_size[0], 0))
        draw = ImageDraw.Draw(canvas)
        footer_lines = [
            f"id={row['image_id']} split={row['split']} class={row['task_pattern_3class']} binary={row['task_binary']}",
            f"status={row['annotation_status']} repair={row['repair_action']}",
            f"interpretation={row['supervision_interpretation']}",
        ]
        for line_index, text in enumerate(footer_lines):
            draw.text((10, tile_size[1] + 8 + (line_index * 18)), text, fill=(0, 0, 0))
        canvas.save(output_dir / f"{row['image_id']}_{row['annotation_status']}.png")


def _build_report(rows: list[dict[str, object]]) -> str:
    by_class = _summary_rows(rows, ("task_pattern_3class", "annotation_status"))
    by_split = _summary_rows(rows, ("split", "annotation_status"))
    total = len(rows)
    mask_count = sum(1 for row in rows if bool(row["has_ulcer_mask"]))
    candidate_positive_missing = sum(1 for row in rows if bool(row["candidate_positive_missing"]))
    candidate_negative = sum(1 for row in rows if bool(row["candidate_negative"]))
    uncertain = sum(1 for row in rows if bool(row["uncertain_semantics"]))

    lines = [
        "# Ulcer-Mask Supervision Audit",
        "",
        "## Semantics Finding",
        "",
        "- Historical ulcer-mask coverage is not label-complete for all ulcer-present images.",
        f"- Total audited images: {total}",
        f"- Images with historical ulcer masks: {mask_count}",
        f"- Candidate positive cases missing lesion masks: {candidate_positive_missing}",
        f"- Candidate negative cases needing explicit empty-mask review: {candidate_negative}",
        f"- Rows flagged with uncertain semantics under the historical task: {uncertain}",
        "",
        "## Class Summary",
        "",
        "| class | status | count |",
        "| --- | --- | ---: |",
    ]
    for row in by_class:
        lines.append(f"| {row['task_pattern_3class']} | {row['annotation_status']} | {row['count']} |")
    lines.extend(
        [
            "",
            "## Split Summary",
            "",
            "| split | status | count |",
            "| --- | --- | ---: |",
        ]
    )
    for row in by_split:
        lines.append(f"| {row['split']} | {row['annotation_status']} | {row['count']} |")
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- Historical ulcer masks are incomplete auxiliary supervision.",
            "- Missing masks are not negatives.",
            "- Canonical severity work must not treat mask absence as a negative label.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    manifest_rows = _load_csv_rows(args.manifest)
    split_rows = _load_csv_rows(args.split_file)
    rows = _build_audit_rows(manifest_rows, _split_lookup(split_rows))
    write_csv_rows(output_root / "ulcer_supervision_audit.csv", rows)
    write_text(output_root / "summary.md", _build_report(rows))
    _render_review_panels(_review_rows(rows), output_root / "review_panels", args.render_limit)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
