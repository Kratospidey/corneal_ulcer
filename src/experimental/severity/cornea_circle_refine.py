from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CorneaCircleFit:
    center_x: float
    center_y: float
    radius: float
    area_pixels: int
    fill_ratio: float
    mean_abs_residual: float
    residual_std: float
    confidence: float
    quality: str
    fallback_used: bool
    source: str

    def to_feature_dict(self, image_width: int, image_height: int) -> dict[str, float | int | str | bool]:
        short_side = float(max(1, min(image_width, image_height)))
        return {
            "cornea_center_x_norm": float(self.center_x / max(1.0, float(image_width))),
            "cornea_center_y_norm": float(self.center_y / max(1.0, float(image_height))),
            "cornea_radius_norm": float(self.radius / short_side),
            "cornea_area_fraction": float(self.area_pixels / max(1.0, float(image_width * image_height))),
            "cornea_fill_ratio": float(self.fill_ratio),
            "cornea_fit_mean_abs_residual_norm": float(self.mean_abs_residual / max(self.radius, 1e-6)),
            "cornea_fit_residual_std_norm": float(self.residual_std / max(self.radius, 1e-6)),
            "cornea_fit_confidence": float(self.confidence),
            "cornea_fit_quality": self.quality,
            "cornea_fit_fallback_used": bool(self.fallback_used),
            "cornea_fit_source": self.source,
        }


def _mask_edge_points(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Expected a 2D mask.")
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.float64)

    interior = mask.copy()
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    edge = mask & ~interior
    ys, xs = np.nonzero(edge)
    if xs.size == 0:
        ys, xs = np.nonzero(mask)
    return np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))


def _equivalent_circle(mask: np.ndarray) -> CorneaCircleFit:
    ys, xs = np.nonzero(mask)
    area_pixels = int(mask.sum())
    if xs.size == 0:
        height, width = mask.shape
        radius = float(min(height, width) / 4.0)
        center_x = float(width / 2.0)
        center_y = float(height / 2.0)
        return CorneaCircleFit(
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            area_pixels=0,
            fill_ratio=0.0,
            mean_abs_residual=radius,
            residual_std=radius,
            confidence=0.0,
            quality="failed",
            fallback_used=True,
            source="image_center",
        )

    center_x = float(xs.mean())
    center_y = float(ys.mean())
    radius = float(math.sqrt(max(area_pixels, 1) / math.pi))
    fill_ratio = float(area_pixels / max(math.pi * radius * radius, 1e-6))
    return CorneaCircleFit(
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        area_pixels=area_pixels,
        fill_ratio=fill_ratio,
        mean_abs_residual=0.0,
        residual_std=0.0,
        confidence=0.35,
        quality="fallback",
        fallback_used=True,
        source="equivalent_area",
    )


def fit_cornea_circle(mask: np.ndarray) -> CorneaCircleFit:
    mask_bool = np.asarray(mask, dtype=bool)
    area_pixels = int(mask_bool.sum())
    if area_pixels < 32:
        return _equivalent_circle(mask_bool)

    edge_points = _mask_edge_points(mask_bool)
    if edge_points.shape[0] < 8:
        return _equivalent_circle(mask_bool)

    x = edge_points[:, 0]
    y = edge_points[:, 1]
    design = np.column_stack((x, y, np.ones_like(x)))
    target = -(x * x + y * y)

    try:
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return _equivalent_circle(mask_bool)

    a_param, b_param, c_param = [float(value) for value in solution]
    center_x = -a_param / 2.0
    center_y = -b_param / 2.0
    radius_sq = (center_x * center_x) + (center_y * center_y) - c_param
    if not np.isfinite(radius_sq) or radius_sq <= 1.0:
        return _equivalent_circle(mask_bool)
    radius = float(math.sqrt(radius_sq))

    radial_distances = np.sqrt(((x - center_x) ** 2) + ((y - center_y) ** 2))
    residuals = radial_distances - radius
    mean_abs_residual = float(np.mean(np.abs(residuals)))
    residual_std = float(np.std(residuals))
    fill_ratio = float(area_pixels / max(math.pi * radius * radius, 1e-6))

    residual_norm = mean_abs_residual / max(radius, 1e-6)
    fill_penalty = abs(fill_ratio - 1.0)
    confidence = float(max(0.0, 1.0 - min(1.0, (4.0 * residual_norm) + (0.5 * fill_penalty))))
    if confidence >= 0.82:
        quality = "high"
    elif confidence >= 0.6:
        quality = "medium"
    else:
        quality = "low"

    if not np.isfinite(center_x) or not np.isfinite(center_y) or not np.isfinite(radius):
        return _equivalent_circle(mask_bool)

    return CorneaCircleFit(
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        area_pixels=area_pixels,
        fill_ratio=fill_ratio,
        mean_abs_residual=mean_abs_residual,
        residual_std=residual_std,
        confidence=confidence,
        quality=quality,
        fallback_used=False,
        source="mask_least_squares",
    )
