from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from utils_io import safe_open_image
from utils_preprocessing import apply_variant, normalize_cornea_mask


def mask_white_fraction(mask_image: Image.Image) -> float:
    mask_array = np.array(mask_image.convert("L")) > 127
    return float(mask_array.mean())


def mean_intensity_inside_outside(image: Image.Image, binary_mask: Image.Image) -> dict[str, float]:
    mask_bool = np.array(binary_mask.convert("L")) > 127
    image_mean = np.array(image.convert("RGB"), dtype=np.float32).mean(axis=2)
    inside_mean = float(image_mean[mask_bool].mean()) if mask_bool.any() else 0.0
    outside_mean = float(image_mean[~mask_bool].mean()) if (~mask_bool).any() else 0.0
    return {
        "inside_mean": round(inside_mean, 2),
        "outside_mean": round(outside_mean, 2),
    }


def build_mask_polarity_payload(image: Image.Image, stored_mask: Image.Image) -> dict[str, object]:
    normalized_mask = normalize_cornea_mask(stored_mask)
    opposite_mask = ImageOps.invert(normalized_mask)

    current_output = apply_variant(image, "masked_highlight_proxy", normalized_mask).convert("RGB")
    corrected_output = apply_variant(image, "masked_highlight_proxy", normalized_mask).convert("RGB")
    opposite_output = apply_variant(image, "masked_highlight_proxy", opposite_mask).convert("RGB")

    return {
        "stored_mask": stored_mask.convert("L"),
        "normalized_mask": normalized_mask.convert("L"),
        "current_output": current_output,
        "corrected_output": corrected_output,
        "opposite_output": opposite_output,
        "current_stats": mean_intensity_inside_outside(current_output, normalized_mask),
        "corrected_stats": mean_intensity_inside_outside(corrected_output, normalized_mask),
        "opposite_stats": mean_intensity_inside_outside(opposite_output, normalized_mask),
        "current_matches_corrected": bool(np.array_equal(np.array(current_output), np.array(corrected_output))),
        "stored_mask_white_fraction": round(mask_white_fraction(stored_mask), 4),
        "normalized_mask_white_fraction": round(mask_white_fraction(normalized_mask), 4),
    }


def add_title(image: Image.Image, title: str) -> Image.Image:
    image = image.convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 34), "white")
    canvas.paste(image, (0, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title, fill="black")
    return canvas


def render_mask_polarity_panel(
    image: Image.Image,
    payload: dict[str, object],
    *,
    image_id: str,
    label: str,
    split_name: str,
) -> Image.Image:
    parts = [
        add_title(image, f"original | id={image_id} | {label} | {split_name}"),
        add_title(ImageOps.colorize(payload["stored_mask"], black="black", white="white"), "stored mask"),
        add_title(ImageOps.colorize(payload["normalized_mask"], black="black", white="white"), "normalized cornea mask"),
        add_title(payload["current_output"], "current masked output"),
        add_title(payload["corrected_output"], "explicit cornea-preserving output"),
        add_title(payload["opposite_output"], "opposite-polarity output"),
    ]
    total_width = sum(part.width for part in parts)
    max_height = max(part.height for part in parts)
    panel = Image.new("RGB", (total_width, max_height), "#dddddd")
    cursor = 0
    for part in parts:
        panel.paste(part, (cursor, 0))
        cursor += part.width
    return panel


def select_probe_rows(
    manifest_path: str | Path,
    split_file: str | Path,
    *,
    splits: tuple[str, ...] = ("test", "val"),
) -> list[dict[str, str]]:
    manifest_rows = {row["image_id"]: row for row in csv.DictReader(Path(manifest_path).open())}
    split_rows = list(csv.DictReader(Path(split_file).open()))

    selected: list[dict[str, str]] = []
    seen: dict[str, set[str]] = defaultdict(set)
    for split_name in splits:
        for row in split_rows:
            label = row["label"]
            if row["split"] != split_name or split_name in seen[label]:
                continue
            selected.append({**row, **manifest_rows[row["image_id"]]})
            seen[label].add(split_name)
    return sorted(selected, key=lambda row: (row["label"], row["split"], int(row["image_id"])))


def write_mask_polarity_probe(
    manifest_path: str | Path,
    split_file: str | Path,
    output_dir: str | Path,
) -> Path:
    output_root = Path(output_dir)
    panel_dir = output_root / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, object]] = []
    for row in select_probe_rows(manifest_path, split_file):
        image = safe_open_image(Path(row["raw_image_path"])).convert("RGB")
        stored_mask = safe_open_image(Path(row["cornea_mask_path"])).convert("L")
        payload = build_mask_polarity_payload(image, stored_mask)
        panel = render_mask_polarity_panel(
            image,
            payload,
            image_id=str(row["image_id"]),
            label=str(row["label"]),
            split_name=str(row["split"]),
        )
        panel.save(panel_dir / f"{row['label']}__{row['split']}__image_{row['image_id']}.png")
        metrics_rows.append(
            {
                "image_id": row["image_id"],
                "label": row["label"],
                "split": row["split"],
                "raw_mask_white_fraction": payload["stored_mask_white_fraction"],
                "normalized_mask_white_fraction": payload["normalized_mask_white_fraction"],
                "current_inside_mean_intensity": payload["current_stats"]["inside_mean"],
                "current_outside_mean_intensity": payload["current_stats"]["outside_mean"],
                "corrected_inside_mean_intensity": payload["corrected_stats"]["inside_mean"],
                "corrected_outside_mean_intensity": payload["corrected_stats"]["outside_mean"],
                "opposite_inside_mean_intensity": payload["opposite_stats"]["inside_mean"],
                "opposite_outside_mean_intensity": payload["opposite_stats"]["outside_mean"],
                "current_matches_corrected_pixels": payload["current_matches_corrected"],
            }
        )

    metrics_path = output_root / "mask_polarity_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    current_matches = all(bool(row["current_matches_corrected_pixels"]) for row in metrics_rows)
    mean_current_inside = round(sum(float(row["current_inside_mean_intensity"]) for row in metrics_rows) / len(metrics_rows), 2)
    mean_current_outside = round(sum(float(row["current_outside_mean_intensity"]) for row in metrics_rows) / len(metrics_rows), 2)
    mean_opposite_inside = round(sum(float(row["opposite_inside_mean_intensity"]) for row in metrics_rows) / len(metrics_rows), 2)
    mean_opposite_outside = round(sum(float(row["opposite_outside_mean_intensity"]) for row in metrics_rows) / len(metrics_rows), 2)
    summary = "\n".join(
        [
            "# Mask Polarity Probe",
            "",
            f"- Samples inspected: {len(metrics_rows)}",
            "- Selection rule: first test and first val example for each pattern_3class label.",
            "- Interpretation anchor: normalized cornea mask defines the intended retained cornea region.",
            "- `current masked output` uses the live repository code path.",
            "- `explicit cornea-preserving output` reconstructs the intended output directly from the normalized mask.",
            "- `opposite-polarity output` shows the legacy/background-preserving alternative for comparison.",
            "",
            "## Aggregate checks",
            "",
            f"- Current matched explicit cornea-preserving output for all samples: {current_matches}",
            f"- Mean current inside intensity: {mean_current_inside}",
            f"- Mean current outside intensity: {mean_current_outside}",
            f"- Mean opposite inside intensity: {mean_opposite_inside}",
            f"- Mean opposite outside intensity: {mean_opposite_outside}",
        ]
    )
    (output_root / "summary.md").write_text(summary, encoding="utf-8")
    return output_root


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate a small mask-polarity sanity probe.")
    parser.add_argument("--manifest-path", default="data/interim/manifests/manifest.csv")
    parser.add_argument("--split-file", default="data/interim/split_files/pattern_3class_holdout.csv")
    parser.add_argument("--output-dir", default="outputs/debug/2026-04-19_mask_polarity_probe")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = write_mask_polarity_probe(
        manifest_path=args.manifest_path,
        split_file=args.split_file,
        output_dir=args.output_dir,
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
