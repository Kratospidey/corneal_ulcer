from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import hashlib
import logging

from utils_io import safe_open_image


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path, hash_size: int = 8) -> str:
    try:
        import imagehash  # type: ignore
    except ImportError:
        imagehash = None

    image = safe_open_image(path)
    if imagehash is not None:
        return str(imagehash.average_hash(image, hash_size=hash_size))

    from PIL import ImageOps  # type: ignore

    gray = ImageOps.grayscale(image).resize((hash_size, hash_size))
    pixels = list(gray.getdata())
    avg = sum(pixels) / max(1, len(pixels))
    return "".join("1" if pixel >= avg else "0" for pixel in pixels)


def build_duplicate_rows(manifest_rows: list[dict[str, Any]], logger: logging.Logger) -> list[dict[str, Any]]:
    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    perceptual_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in manifest_rows:
        raw_path = Path(str(row["raw_image_path"]))
        if not raw_path.exists():
            continue
        try:
            exact_groups[md5_file(raw_path)].append(row)
            perceptual_groups[average_hash(raw_path)].append(row)
        except Exception as exc:
            logger.warning("Skipping duplicate check for %s: %s", row["image_id"], exc)

    duplicates: list[dict[str, Any]] = []
    for group_type, groups in (("exact", exact_groups), ("perceptual", perceptual_groups)):
        for group_id, members in groups.items():
            if len(members) < 2:
                continue
            label_tuples = {
                (
                    str(member["task_pattern_3class"]),
                    str(member["task_severity_5class"]),
                    str(member["task_tg_5class"]),
                )
                for member in members
            }
            for member in members:
                duplicates.append(
                    {
                        "group_type": group_type,
                        "group_id": group_id,
                        "group_size": len(members),
                        "image_id": member["image_id"],
                        "task_pattern_3class": member["task_pattern_3class"],
                        "task_severity_5class": member["task_severity_5class"],
                        "task_tg_5class": member["task_tg_5class"],
                        "cross_label_suspicion": len(label_tuples) > 1,
                    }
                )
    return duplicates
