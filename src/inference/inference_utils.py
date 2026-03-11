from __future__ import annotations

from pathlib import Path

from data.transforms import build_eval_transform
from preprocessing.paper_preprocessing import load_cornea_mask, prepare_paper_image
from preprocessing.raw_preprocessing import prepare_raw_image
from utils_io import safe_open_image


def load_image_for_inference(image_path: str | Path, preprocessing_mode: str = "raw_rgb", cornea_mask_path: str | Path | None = None, image_size: int = 224):
    image = safe_open_image(Path(image_path))
    if preprocessing_mode == "raw_rgb":
        image = prepare_raw_image(image)
    else:
        cornea_mask = load_cornea_mask(cornea_mask_path) if cornea_mask_path else None
        image = prepare_paper_image(image, preprocessing_mode=preprocessing_mode, cornea_mask=cornea_mask)
    transform = build_eval_transform(image_size=image_size)
    return transform(image).unsqueeze(0)
