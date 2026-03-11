from __future__ import annotations

from pathlib import Path
import logging

from utils_io import safe_open_image


def mask_coverage(mask_image) -> float:
    gray = mask_image.convert("L")
    histogram = gray.histogram()
    total = sum(histogram) or 1
    foreground = total - histogram[0]
    return foreground / total


def overlay_mask(base_image, mask_image, color=(0, 255, 0), alpha: int = 100):
    from PIL import Image  # type: ignore

    base = base_image.convert("RGBA")
    mask = mask_image.convert("L")
    overlay = Image.new("RGBA", base.size, color + (0,))
    overlay.putalpha(mask.point(lambda pixel: alpha if pixel > 0 else 0))
    return Image.alpha_composite(base, overlay)


def summarize_masks(manifest_rows: list[dict[str, object]], logger: logging.Logger) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in manifest_rows:
        if not row.get("has_cornea_mask"):
            continue
        try:
            cornea_mask = safe_open_image(Path(str(row["cornea_mask_path"])))
        except Exception as exc:
            logger.warning("Skipping cornea mask for %s: %s", row["image_id"], exc)
            continue
        cornea_coverage = mask_coverage(cornea_mask)
        ulcer_coverage = 0.0
        ulcer_ratio = 0.0
        if row.get("has_ulcer_mask"):
            try:
                ulcer_mask = safe_open_image(Path(str(row["ulcer_mask_path"])))
                ulcer_coverage = mask_coverage(ulcer_mask)
                ulcer_ratio = ulcer_coverage / max(cornea_coverage, 1e-6)
            except Exception as exc:
                logger.warning("Skipping ulcer mask for %s: %s", row["image_id"], exc)
        rows.append(
            {
                "image_id": row["image_id"],
                "task_pattern_3class": row["task_pattern_3class"],
                "task_severity_5class": row["task_severity_5class"],
                "task_tg_5class": row["task_tg_5class"],
                "cornea_mask_coverage": round(cornea_coverage, 6),
                "ulcer_mask_coverage": round(ulcer_coverage, 6),
                "ulcer_to_cornea_ratio": round(ulcer_ratio, 6),
            }
        )
    return rows
