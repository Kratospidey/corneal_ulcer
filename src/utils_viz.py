from __future__ import annotations

from pathlib import Path
from typing import Sequence
import logging


def matplotlib_ready() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def save_bar_chart(rows: list[dict[str, object]], label_key: str, value_key: str, title: str, output_path: Path, logger: logging.Logger) -> bool:
    if not rows or not matplotlib_ready():
        return False
    import matplotlib.pyplot as plt  # type: ignore

    labels = [str(row[label_key]) for row in rows]
    values = [float(row[value_key]) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.bar(labels, values)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved figure %s", output_path)
    return True


def save_histogram(values: list[float], title: str, output_path: Path, logger: logging.Logger, bins: int = 30) -> bool:
    if not values or not matplotlib_ready():
        return False
    import matplotlib.pyplot as plt  # type: ignore

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.hist(values, bins=bins)
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved figure %s", output_path)
    return True


def save_scatter_plot(
    rows: list[dict[str, object]],
    x_key: str,
    y_key: str,
    color_key: str,
    title: str,
    output_path: Path,
    logger: logging.Logger,
) -> bool:
    if not rows or not matplotlib_ready():
        return False
    import matplotlib.pyplot as plt  # type: ignore

    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row[color_key]), []).append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 6))
    for label, members in sorted(groups.items()):
        axis.scatter(
            [float(member[x_key]) for member in members],
            [float(member[y_key]) for member in members],
            s=24,
            alpha=0.8,
            label=label,
        )
    axis.set_title(title)
    axis.set_xlabel(x_key)
    axis.set_ylabel(y_key)
    axis.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved figure %s", output_path)
    return True


def save_montage(
    images: list[tuple[str, object]],
    output_path: Path,
    logger: logging.Logger,
    tile_size: tuple[int, int] = (224, 224),
    columns: int = 4,
) -> bool:
    if not images:
        return False
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError:
        return False

    columns = min(columns, max(1, len(images)))
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        resized = image.convert("RGB").resize(tile_size)
        x = (index % columns) * tile_size[0]
        y = (index // columns) * tile_size[1]
        canvas.paste(resized, (x, y))
        draw.text((x + 6, y + 6), str(label), fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    logger.info("Saved montage %s", output_path)
    return True


def save_comparison_grid(
    rows: Sequence[dict[str, object]],
    output_path: Path,
    logger: logging.Logger,
    tile_size: tuple[int, int] = (180, 180),
) -> bool:
    if not rows:
        return False
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError:
        return False

    variants = [str(key) for key in rows[0].keys() if key != "image_id"]
    if not variants:
        return False

    header_height = 28
    columns = len(variants)
    row_count = len(rows)
    canvas = Image.new(
        "RGB",
        (columns * tile_size[0], header_height + row_count * tile_size[1]),
        (8, 8, 8),
    )
    draw = ImageDraw.Draw(canvas)

    for column_index, variant_name in enumerate(variants):
        draw.text((column_index * tile_size[0] + 6, 6), variant_name, fill=(255, 255, 255))

    for row_index, row in enumerate(rows):
        for column_index, variant_name in enumerate(variants):
            image = row[variant_name]
            if image is None:
                continue
            resized = image.convert("RGB").resize(tile_size)
            x = column_index * tile_size[0]
            y = header_height + row_index * tile_size[1]
            canvas.paste(resized, (x, y))
            draw.text((x + 6, y + 6), str(row["image_id"]), fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    logger.info("Saved comparison grid %s", output_path)
    return True
