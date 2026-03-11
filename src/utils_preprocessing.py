from __future__ import annotations

from pathlib import Path
from typing import Callable
import logging

from utils_io import safe_open_image


def otsu_threshold(gray_image):
    histogram = gray_image.histogram()
    total = sum(histogram)
    sum_total = sum(index * count for index, count in enumerate(histogram))
    sum_background = 0.0
    weight_background = 0.0
    best_threshold = 0
    best_variance = -1.0
    for threshold, count in enumerate(histogram):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += threshold * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def apply_variant(image, variant_name: str, cornea_mask=None):
    from PIL import Image, ImageFilter, ImageOps  # type: ignore

    rgb = image.convert("RGB")
    if variant_name == "raw_rgb":
        return rgb
    if variant_name == "blue_channel_removed":
        red, green, blue = rgb.split()
        black = blue.point(lambda _: 0)
        return Image.merge("RGB", (red, green, black))
    if variant_name == "grayscale":
        return ImageOps.grayscale(rgb)
    if variant_name == "gaussian_blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=2.0))
    if variant_name == "otsu_threshold":
        gray = ImageOps.grayscale(rgb)
        threshold = otsu_threshold(gray)
        return gray.point(lambda pixel: 255 if pixel >= threshold else 0)
    if variant_name == "masked_highlight_proxy":
        red, green, blue = rgb.split()
        proxy = green.point(lambda pixel: min(255, int(pixel * 1.25)))
        composite = Image.merge("RGB", (red.point(lambda pixel: int(pixel * 0.6)), proxy, blue.point(lambda pixel: int(pixel * 0.35))))
        if cornea_mask is not None:
            mask = cornea_mask.convert("L")
            background = Image.new("RGB", rgb.size, (0, 0, 0))
            return Image.composite(composite, background, mask)
        return composite
    if variant_name == "clahe_exploratory":
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return rgb
        array = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2LAB)
        lab_l, lab_a, lab_b = cv2.split(array)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        merged = cv2.merge((clahe.apply(lab_l), lab_a, lab_b))
        corrected = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(corrected)
    raise ValueError(f"Unknown preprocessing variant: {variant_name}")


def available_variants() -> list[str]:
    return [
        "raw_rgb",
        "blue_channel_removed",
        "grayscale",
        "gaussian_blur",
        "otsu_threshold",
        "masked_highlight_proxy",
        "clahe_exploratory",
    ]


def sample_variant_images(
    sample_rows: list[dict[str, str]],
    variant_name: str,
    logger: logging.Logger,
) -> list[tuple[str, object]]:
    images: list[tuple[str, object]] = []
    for row in sample_rows:
        try:
            image = safe_open_image(Path(row["raw_image_path"]))
            cornea_mask = None
            if row.get("cornea_mask_path"):
                try:
                    cornea_mask = safe_open_image(Path(row["cornea_mask_path"]))
                except Exception:
                    cornea_mask = None
            images.append((row["image_id"], apply_variant(image, variant_name, cornea_mask)))
        except Exception as exc:
            logger.warning("Skipping preprocessing sample for %s: %s", row.get("image_id"), exc)
    return images
