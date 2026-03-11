from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import csv
import json
import logging
import os


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("eda")


def ensure_directories(paths: Iterable[Path | str]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def safe_read_text(path: Path | str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8")


def write_text(path: Path | str, content: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def write_json(path: Path | str, payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv_rows(path: Path | str, rows: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        file_path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def try_write_parquet(path: Path | str, rows: list[dict[str, Any]], logger: logging.Logger) -> bool:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        logger.warning("Skipping Parquet export because pandas is not installed.")
        return False

    try:
        pd.DataFrame(rows).to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning("Skipping Parquet export at %s: %s", path, exc)
        return False
    return True


def safe_open_image(path: Path | str):
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime optional dependency
        raise RuntimeError("Pillow is required for image operations.") from exc

    image_path = Path(path)
    with Image.open(image_path) as image:
        image.load()
        return image.copy()


def markdown_section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def flatten_counter(counter: dict[str, Any], key_name: str, value_name: str) -> list[dict[str, Any]]:
    return [{key_name: key, value_name: value} for key, value in counter.items()]


def relative_to_workspace(path: Path | str) -> str:
    return os.fspath(Path(path))
