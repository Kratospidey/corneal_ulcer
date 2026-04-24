from __future__ import annotations

from pathlib import Path

from utils_io import safe_open_image
from utils_preprocessing import apply_variant


def prepare_paper_image(image, preprocessing_mode: str = "raw_rgb", cornea_mask=None):
    return apply_variant(image, preprocessing_mode, cornea_mask)


def load_cornea_mask(mask_path: str | Path | None):
    if not mask_path:
        return None
    return safe_open_image(Path(mask_path))
