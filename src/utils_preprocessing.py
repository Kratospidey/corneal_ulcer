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


def normalize_binary_mask(mask_image, invert_if_white_corners: bool = True):
    from PIL import ImageOps  # type: ignore

    mask = mask_image.convert("L")
    if invert_if_white_corners:
        corner_coordinates = (
            (0, 0),
            (max(0, mask.width - 1), 0),
            (0, max(0, mask.height - 1)),
            (max(0, mask.width - 1), max(0, mask.height - 1)),
        )
        corner_mean = sum(int(mask.getpixel(coord)) for coord in corner_coordinates) / len(corner_coordinates)
        if corner_mean > 127:
            mask = ImageOps.invert(mask)
    return mask.point(lambda pixel: 255 if pixel > 127 else 0)


def normalize_cornea_mask(mask_image):
    return normalize_binary_mask(mask_image, invert_if_white_corners=True)


def normalize_ulcer_mask(mask_image):
    return normalize_binary_mask(mask_image, invert_if_white_corners=True)


def extract_cornea_roi_with_context(image, cornea_mask, context_ratio: float = 0.12):
    if cornea_mask is None:
        return image.convert("RGB")

    mask = cornea_mask.convert("L")
    bbox = mask.getbbox()
    if bbox is None:
        return image.convert("RGB")

    left, top, right, bottom = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    side = int(round(max(width, height) * (1.0 + float(context_ratio) * 2.0)))
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0

    crop_left = max(0, int(round(center_x - side / 2.0)))
    crop_top = max(0, int(round(center_y - side / 2.0)))
    crop_right = min(image.width, crop_left + side)
    crop_bottom = min(image.height, crop_top + side)

    if crop_right - crop_left < side:
        crop_left = max(0, crop_right - side)
    if crop_bottom - crop_top < side:
        crop_top = max(0, crop_bottom - side)

    return image.convert("RGB").crop((crop_left, crop_top, crop_right, crop_bottom))


def _extract_cornea_square_crop(
    image,
    cornea_mask,
    *,
    context_ratio: float,
    min_side_ratio: float | None = None,
    max_side_ratio: float | None = None,
):
    rgb = image.convert("RGB")
    if cornea_mask is None:
        return rgb

    bbox = cornea_mask.convert("L").getbbox()
    if bbox is None:
        return rgb

    left, top, right, bottom = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    side = float(max(width, height)) * (1.0 + float(context_ratio) * 2.0)
    short_side = float(min(rgb.width, rgb.height))
    if min_side_ratio is not None:
        side = max(side, short_side * float(min_side_ratio))
    if max_side_ratio is not None:
        side = min(side, short_side * float(max_side_ratio))
    side = max(1, int(round(side)))

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_left = int(round(center_x - side / 2.0))
    crop_top = int(round(center_y - side / 2.0))
    crop_right = crop_left + side
    crop_bottom = crop_top + side
    return rgb.crop((crop_left, crop_top, crop_right, crop_bottom))


def extract_cornea_crop_ctx_v1(image, cornea_mask):
    return _extract_cornea_square_crop(
        image,
        cornea_mask,
        context_ratio=0.28,
        max_side_ratio=0.98,
    )


def extract_cornea_crop_scale_v1(image, cornea_mask):
    return _extract_cornea_square_crop(
        image,
        cornea_mask,
        context_ratio=0.18,
        min_side_ratio=0.72,
        max_side_ratio=0.98,
    )


def apply_variant(image, variant_name: str, cornea_mask=None):
    from PIL import Image, ImageChops, ImageFilter, ImageOps  # type: ignore

    rgb = image.convert("RGB")
    if variant_name == "raw_rgb":
        return rgb
    if variant_name == "cornea_roi_context":
        return extract_cornea_roi_with_context(rgb, cornea_mask)
    if variant_name == "cornea_crop_ctx_v1":
        return extract_cornea_crop_ctx_v1(rgb, cornea_mask)
    if variant_name == "cornea_crop_scale_v1":
        return extract_cornea_crop_scale_v1(rgb, cornea_mask)
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
    if variant_name == "diagnostics_green_mask":
        red, green, blue = rgb.split()
        blue_removed = Image.merge("RGB", (red, green, blue.point(lambda _: 0)))
        gray = ImageOps.grayscale(blue_removed)
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=2.0))
        threshold = otsu_threshold(blurred)
        otsu_mask = blurred.point(lambda pixel: 255 if pixel >= threshold else 0)
        if cornea_mask is not None:
            otsu_mask = ImageChops.multiply(otsu_mask, cornea_mask.convert("L"))
        green_only = Image.merge("RGB", (Image.new("L", rgb.size, 0), gray, Image.new("L", rgb.size, 0)))
        return Image.composite(green_only, Image.new("RGB", rgb.size, (0, 0, 0)), otsu_mask)
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
        "cornea_roi_context",
        "cornea_crop_ctx_v1",
        "cornea_crop_scale_v1",
        "blue_channel_removed",
        "grayscale",
        "gaussian_blur",
        "otsu_threshold",
        "masked_highlight_proxy",
        "diagnostics_green_mask",
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
                    cornea_mask = normalize_cornea_mask(safe_open_image(Path(row["cornea_mask_path"])))
                except Exception:
                    cornea_mask = None
            images.append((row["image_id"], apply_variant(image, variant_name, cornea_mask)))
        except Exception as exc:
            logger.warning("Skipping preprocessing sample for %s: %s", row.get("image_id"), exc)
    return images
