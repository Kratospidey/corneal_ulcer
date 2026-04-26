from __future__ import annotations

from pathlib import Path
import logging

from utils_io import safe_open_image


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

    bbox = normalize_cornea_mask(cornea_mask).getbbox()
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


def extract_cornea_crop_scale_v1(image, cornea_mask):
    return _extract_cornea_square_crop(
        image,
        cornea_mask,
        context_ratio=0.18,
        min_side_ratio=0.72,
        max_side_ratio=0.98,
    )


def extract_cornea_crop_slightly_tight(image, cornea_mask):
    return _extract_cornea_square_crop(
        image,
        cornea_mask,
        context_ratio=0.10,
        min_side_ratio=0.65,
        max_side_ratio=0.90,
    )


def extract_cornea_crop_slightly_wide(image, cornea_mask):
    return _extract_cornea_square_crop(
        image,
        cornea_mask,
        context_ratio=0.25,
        min_side_ratio=0.75,
        max_side_ratio=1.00,
    )

def extract_cornea_crop_wide_context_v1(image, cornea_mask):
    return _extract_cornea_square_crop(
        image,
        cornea_mask,
        context_ratio=0.45,
        min_side_ratio=0.88,
        max_side_ratio=1.00,
    )

def _extract_shifted_cornea_crop(image, cornea_mask, x_shift_ratio: float, y_shift_ratio: float):
    rgb = image.convert("RGB")
    if cornea_mask is None:
        return rgb

    bbox = normalize_cornea_mask(cornea_mask).getbbox()
    if bbox is None:
        return rgb

    left, top, right, bottom = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    side = float(max(width, height)) * (1.0 + 0.18 * 2.0)
    short_side = float(min(rgb.width, rgb.height))
    side = max(side, short_side * 0.72)
    side = min(side, short_side * 0.98)
    side = max(1, int(round(side)))

    center_x = (left + right) / 2.0 + (x_shift_ratio * side)
    center_y = (top + bottom) / 2.0 + (y_shift_ratio * side)
    crop_left = int(round(center_x - side / 2.0))
    crop_top = int(round(center_y - side / 2.0))
    crop_right = crop_left + side
    crop_bottom = crop_top + side
    return rgb.crop((crop_left, crop_top, crop_right, crop_bottom))


def apply_variant(image, variant_name: str, cornea_mask=None):
    from PIL import Image  # type: ignore

    rgb = image.convert("RGB")
    if variant_name == "raw_rgb":
        return rgb
    if variant_name == "cornea_crop_scale_v1":
        return extract_cornea_crop_scale_v1(rgb, cornea_mask)
    if variant_name == "cornea_crop_slightly_tight":
        return extract_cornea_crop_slightly_tight(rgb, cornea_mask)
    if variant_name == "cornea_crop_slightly_wide":
        return extract_cornea_crop_slightly_wide(rgb, cornea_mask)
    if variant_name == "cornea_crop_wide_context_v1":
        return extract_cornea_crop_wide_context_v1(rgb, cornea_mask)
    if variant_name == "shift_left_up":
        return _extract_shifted_cornea_crop(rgb, cornea_mask, x_shift_ratio=-0.05, y_shift_ratio=-0.05)
    if variant_name == "shift_right_down":
        return _extract_shifted_cornea_crop(rgb, cornea_mask, x_shift_ratio=0.05, y_shift_ratio=0.05)
    if variant_name == "shift_left_down":
        return _extract_shifted_cornea_crop(rgb, cornea_mask, x_shift_ratio=-0.05, y_shift_ratio=0.05)
    if variant_name == "shift_right_up":
        return _extract_shifted_cornea_crop(rgb, cornea_mask, x_shift_ratio=0.05, y_shift_ratio=-0.05)
    if variant_name == "crop_scale_raw_multiscale":
        return _extract_cornea_square_crop(
            rgb,
            cornea_mask,
            context_ratio=0.20,
            min_side_ratio=0.72,
            max_side_ratio=0.98,
        )
    raise ValueError(f"Unknown preprocessing variant: {variant_name}")


def available_variants() -> list[str]:
    return [
        "raw_rgb",
        "cornea_crop_scale_v1",
        "cornea_crop_slightly_tight",
        "cornea_crop_slightly_wide",
        "cornea_crop_wide_context_v1",
        "shift_left_up",
        "shift_right_down",
        "shift_left_down",
        "shift_right_up",
        "crop_scale_raw_multiscale",
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
