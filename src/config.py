from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import importlib.util
import os


def dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


@dataclass
class EDAConfig:
    data_root: Path = Path("data/raw/sustech_sysu")
    output_root: Path = Path("outputs")
    interim_root: Path = Path("data/interim")
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    figure_dpi: int = 160
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    montage_sample_count: int = 16
    extreme_sample_count: int = 12
    embedding_backbones: list[str] = field(default_factory=lambda: ["convnextv2_tiny"])
    embedding_compare_variant: str = "masked_highlight_proxy"
    preprocessing_variants: list[str] = field(
        default_factory=lambda: [
            "raw_rgb",
            "blue_channel_removed",
            "grayscale",
            "gaussian_blur",
            "otsu_threshold",
            "masked_highlight_proxy",
            "clahe_exploratory",
        ]
    )
    duplicate_exact_threshold: float = 1.0
    duplicate_perceptual_hamming_threshold: int = 4
    duplicate_embedding_cosine_threshold: float = 0.995
    blur_low_threshold: float = 20.0
    brightness_low_threshold: float = 40.0
    brightness_high_threshold: float = 215.0

    @property
    def manifest_dir(self) -> Path:
        return self.interim_root / "manifests"

    @property
    def cleaned_metadata_dir(self) -> Path:
        return self.interim_root / "cleaned_metadata"

    @property
    def split_dir(self) -> Path:
        return self.interim_root / "split_files"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def cache_dir(self) -> Path:
        return self.output_root / "cache"

    @property
    def embeddings_dir(self) -> Path:
        return self.output_root / "embeddings"

    @property
    def reports_dir(self) -> Path:
        return self.output_root / "reports"

    @property
    def masks_available(self) -> bool:
        return (self.data_root / "corneaLabels").is_dir() or (self.data_root / "ulcerLabels").is_dir()

    @property
    def runtime_features(self) -> dict[str, bool]:
        return {
            "torch": dependency_available("torch"),
            "torchvision": dependency_available("torchvision"),
            "timm": dependency_available("timm"),
            "kornia": dependency_available("kornia"),
            "cv2": dependency_available("cv2"),
            "numpy": dependency_available("numpy"),
            "pandas": dependency_available("pandas"),
            "pyarrow": dependency_available("pyarrow"),
            "sklearn": dependency_available("sklearn"),
            "matplotlib": dependency_available("matplotlib"),
            "PIL": dependency_available("PIL"),
            "imagehash": dependency_available("imagehash"),
            "umap": dependency_available("umap"),
            "yaml": dependency_available("yaml"),
        }


def build_config(
    *,
    data_root: str | os.PathLike[str] | None = None,
    output_root: str | os.PathLike[str] | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    seed: int | None = None,
) -> EDAConfig:
    config = EDAConfig()
    if data_root is not None:
        config.data_root = Path(data_root)
    if output_root is not None:
        config.output_root = Path(output_root)
    if batch_size is not None:
        config.batch_size = batch_size
    if num_workers is not None:
        config.num_workers = num_workers
    if seed is not None:
        config.seed = seed
    return config


def load_yaml_overrides(config: EDAConfig, yaml_path: str | os.PathLike[str] | None) -> EDAConfig:
    if yaml_path is None:
        return config
    yaml_file = Path(yaml_path)
    if not yaml_file.exists() or not dependency_available("yaml"):
        return config

    import yaml  # type: ignore

    payload = yaml.safe_load(yaml_file.read_text()) or {}
    for key, value in payload.items():
        if hasattr(config, key):
            if key in {"data_root", "output_root", "interim_root"} and value is not None:
                value = Path(value)
            setattr(config, key, value)
    return config


def resolve_device(requested: str = "auto") -> str:
    if requested in {"cpu", "cuda"}:
        return requested
    if not dependency_available("torch"):
        return "cpu"

    import torch  # type: ignore

    return "cuda" if torch.cuda.is_available() else "cpu"


def runtime_summary(config: EDAConfig, requested_device: str = "auto") -> dict[str, Any]:
    summary: dict[str, Any] = {
        "data_root": str(config.data_root),
        "output_root": str(config.output_root),
        "seed": config.seed,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "embedding_backbones": list(config.embedding_backbones),
        "embedding_compare_variant": config.embedding_compare_variant,
        "requested_device": requested_device,
        "resolved_device": resolve_device(requested_device),
    }
    summary.update(config.runtime_features)
    return summary
