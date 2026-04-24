from __future__ import annotations

from pathlib import Path
from statistics import mean
import colorsys
import logging
import math

from utils_io import safe_open_image


def _pixel_channel_stats(image):
    from PIL import ImageStat  # type: ignore

    rgb = image.convert("RGB")
    stat = ImageStat.Stat(rgb)
    channel_mean = stat.mean
    channel_std = stat.stddev
    return rgb, channel_mean, channel_std


def _sample_hsv(rgb):
    width, height = rgb.size
    step_x = max(1, width // 64)
    step_y = max(1, height // 64)
    hues: list[float] = []
    saturations: list[float] = []
    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            red, green, blue = rgb.getpixel((x, y))
            hue, saturation, _ = colorsys.rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
            hues.append(hue)
            saturations.append(saturation)
    return mean(hues) if hues else 0.0, mean(saturations) if saturations else 0.0


def _edge_variance(gray):
    from PIL import ImageFilter, ImageStat  # type: ignore

    edges = gray.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    if not stat.var:
        return 0.0
    return float(stat.var[0])


def compute_image_stats(image_path: Path, logger: logging.Logger) -> dict[str, object]:
    try:
        image = safe_open_image(image_path)
    except Exception as exc:
        logger.warning("Unreadable image %s: %s", image_path, exc)
        return {
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "readable": False,
            "error": str(exc),
        }

    from PIL import ImageOps, ImageStat  # type: ignore

    rgb, channel_mean, channel_std = _pixel_channel_stats(image)
    gray = ImageOps.grayscale(rgb)
    gray_stat = ImageStat.Stat(gray)
    brightness = float(gray_stat.mean[0])
    contrast = float(gray_stat.stddev[0])
    entropy = float(gray.entropy())
    blur_proxy = _edge_variance(gray)
    hue_mean, saturation_mean = _sample_hsv(rgb)
    red_mean, green_mean, blue_mean = channel_mean
    green_dominance = float(green_mean / max(1e-6, (red_mean + blue_mean) / 2.0))
    intensity_center = _center_intensity(gray)
    glare_proxy = _glare_proxy(gray)

    return {
        "image_id": image_path.stem,
        "image_path": str(image_path),
        "readable": True,
        "width": rgb.width,
        "height": rgb.height,
        "aspect_ratio": round(rgb.width / rgb.height, 6) if rgb.height else 0.0,
        "filesize_bytes": image_path.stat().st_size,
        "mode": image.mode,
        "mean_r": round(red_mean, 6),
        "mean_g": round(green_mean, 6),
        "mean_b": round(blue_mean, 6),
        "std_r": round(channel_std[0], 6),
        "std_g": round(channel_std[1], 6),
        "std_b": round(channel_std[2], 6),
        "brightness": round(brightness, 6),
        "contrast": round(contrast, 6),
        "entropy": round(entropy, 6),
        "blur_proxy": round(blur_proxy, 6),
        "mean_hue": round(hue_mean, 6),
        "mean_saturation": round(saturation_mean, 6),
        "green_dominance": round(green_dominance, 6),
        "center_intensity": round(intensity_center, 6),
        "specular_highlight_ratio": round(glare_proxy, 6),
    }


def _center_intensity(gray) -> float:
    width, height = gray.size
    left = width // 4
    right = width - left
    top = height // 4
    bottom = height - top
    crop = gray.crop((left, top, right, bottom))
    from PIL import ImageStat  # type: ignore

    return float(ImageStat.Stat(crop).mean[0])


def _glare_proxy(gray) -> float:
    histogram = gray.histogram()
    total = sum(histogram) or 1
    high_intensity = sum(histogram[245:])
    return high_intensity / total


def summarize_small_data_risk(stats_rows: list[dict[str, object]], manifest_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    readable_count = sum(1 for row in stats_rows if row.get("readable"))
    unreadable_count = len(stats_rows) - readable_count
    pattern_counts: dict[str, int] = {}
    for row in manifest_rows:
        pattern_counts[row["task_pattern_3class"]] = pattern_counts.get(row["task_pattern_3class"], 0) + 1

    def imbalance_ratio(counter: dict[str, int]) -> float:
        if not counter:
            return 0.0
        values = list(counter.values())
        return max(values) / max(1, min(values))

    return [
        {"risk": "readability_failures", "value": unreadable_count},
        {"risk": "pattern_imbalance_ratio", "value": round(imbalance_ratio(pattern_counts), 6)},
        {"risk": "ulcer_mask_subset_ratio", "value": round(sum(1 for row in manifest_rows if row["has_ulcer_mask"]) / max(1, len(manifest_rows)), 6)},
    ]
