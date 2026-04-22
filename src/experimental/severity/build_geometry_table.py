from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config_utils import resolve_config  # noqa: E402
from data.label_utils import get_task_definition  # noqa: E402
from data.transforms import build_eval_transform  # noqa: E402
from experimental.severity.cornea_circle_refine import fit_cornea_circle  # noqa: E402
from model_factory import create_model  # noqa: E402
from preprocessing.paper_preprocessing import load_cornea_mask  # noqa: E402
from utils_io import safe_open_image, write_csv_rows, write_json  # noqa: E402
from utils_preprocessing import apply_variant, normalize_cornea_mask  # noqa: E402


SEVERITY_CLASS_NAMES = (
    "no_ulcer",
    "ulcer_leq_25pct",
    "ulcer_leq_50pct",
    "ulcer_geq_75pct",
    "central_ulcer",
)
ZONE_RADII = {
    "central": 0.33,
    "paracentral": 0.66,
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build a SEV-S1 post-hoc geometry feature table.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--repo-root")
    parser.add_argument("--pattern-config")
    parser.add_argument("--pattern-checkpoint")
    parser.add_argument("--pattern-device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def infer_repo_root(manifest_path: Path, repo_root_override: str | None) -> Path:
    if repo_root_override:
        return Path(repo_root_override).resolve()
    resolved = manifest_path.resolve()
    if len(resolved.parents) >= 4:
        return resolved.parents[3]
    return resolved.parent


def resolve_repo_path(repo_root: Path, raw_value: object) -> Path:
    candidate = Path(str(raw_value))
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def normalize_mask_array(mask_image) -> np.ndarray:
    mask = normalize_cornea_mask(mask_image)
    return np.asarray(mask.convert("L"), dtype=np.uint8) > 0


def stable_quantile(values: np.ndarray, quantile: float, default: float = 0.0) -> float:
    if values.size == 0:
        return float(default)
    return float(np.quantile(values, quantile))


def build_green_response(rgb_array: np.ndarray, cornea_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    red = rgb_array[..., 0]
    green = rgb_array[..., 1]
    blue = rgb_array[..., 2]
    response = (green - (0.5 * (red + blue))) / 255.0
    cornea_values = response[cornea_mask]
    median = float(np.median(cornea_values)) if cornea_values.size else 0.0
    mad = float(np.median(np.abs(cornea_values - median))) if cornea_values.size else 0.0
    scale = max(1e-6, 1.4826 * mad)
    z_response = (response - median) / scale
    z_response = np.where(cornea_mask, z_response, 0.0)
    return response, z_response


def build_proxy_masks(z_response: np.ndarray, cornea_mask: np.ndarray) -> dict[str, np.ndarray]:
    masks = {
        "t0_5": (z_response > 0.5) & cornea_mask,
        "t1_0": (z_response > 1.0) & cornea_mask,
        "t1_5": (z_response > 1.5) & cornea_mask,
        "t2_0": (z_response > 2.0) & cornea_mask,
        "t2_5": (z_response > 2.5) & cornea_mask,
        "t3_0": (z_response > 3.0) & cornea_mask,
        "t3_5": (z_response > 3.5) & cornea_mask,
    }
    if masks["t1_0"].sum() == 0 and masks["t0_5"].sum() > 0:
        masks["primary"] = masks["t0_5"]
        masks["primary_threshold"] = np.asarray([0.5], dtype=np.float32)
    else:
        masks["primary"] = masks["t1_0"]
        masks["primary_threshold"] = np.asarray([1.0], dtype=np.float32)
    return masks


def normalized_distance_map(height: int, width: int, center_x: float, center_y: float, radius: float) -> np.ndarray:
    ys, xs = np.indices((height, width), dtype=np.float32)
    dist = np.sqrt(((xs - center_x) ** 2) + ((ys - center_y) ** 2))
    return dist / max(radius, 1e-6)


def component_features(mask: np.ndarray, distance_map: np.ndarray) -> dict[str, float]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for geometry feature extraction.") from exc

    binary = mask.astype(np.uint8)
    area_pixels = float(binary.sum())
    if area_pixels <= 0.0:
        return {
            "lesion_component_count": 0.0,
            "lesion_largest_component_fraction": 0.0,
            "lesion_component_area_std_norm": 0.0,
            "lesion_dispersion_norm": 0.0,
            "lesion_compactness": 0.0,
            "lesion_solidity": 0.0,
            "lesion_bbox_aspect_ratio": 0.0,
            "lesion_eccentricity": 0.0,
        }

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    largest_component_fraction = float(component_areas.max() / max(area_pixels, 1.0))
    area_std_norm = float(component_areas.std() / max(area_pixels, 1.0)) if component_areas.size > 1 else 0.0

    ys, xs = np.nonzero(mask)
    weights = np.ones(xs.shape[0], dtype=np.float32)
    centroid_x = float(np.average(xs, weights=weights))
    centroid_y = float(np.average(ys, weights=weights))
    lesion_distances = distance_map[mask]
    dispersion_norm = float(np.std(lesion_distances)) if lesion_distances.size else 0.0

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        compactness = 0.0
        solidity = 0.0
        aspect_ratio = 0.0
        eccentricity = 0.0
    else:
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        compactness = float((4.0 * math.pi * contour_area) / max(perimeter * perimeter, 1e-6))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(contour_area / max(hull_area, 1e-6))
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = float(width / max(height, 1))
        points = contour.reshape(-1, 2).astype(np.float32)
        if points.shape[0] >= 5:
            covariance = np.cov(points.T)
            eigenvalues = np.sort(np.real(np.linalg.eigvals(covariance)))
            major = float(max(eigenvalues[-1], 1e-6))
            minor = float(max(eigenvalues[0], 0.0))
            eccentricity = float(math.sqrt(max(0.0, 1.0 - (minor / major))))
        else:
            eccentricity = 0.0

    return {
        "lesion_component_count": float(component_count - 1),
        "lesion_largest_component_fraction": largest_component_fraction,
        "lesion_component_area_std_norm": area_std_norm,
        "lesion_dispersion_norm": dispersion_norm,
        "lesion_compactness": compactness,
        "lesion_solidity": solidity,
        "lesion_bbox_aspect_ratio": aspect_ratio,
        "lesion_eccentricity": eccentricity,
        "lesion_centroid_x": centroid_x,
        "lesion_centroid_y": centroid_y,
    }


class PatternScorer:
    def __init__(self, config_path: str, checkpoint_path: str, device_name: str) -> None:
        import torch  # type: ignore

        config = resolve_config(config_path)
        task_config = resolve_config(config["task_config"])
        task_definition = get_task_definition(str(task_config["task_name"]))
        self.class_names = tuple(task_definition.class_names)
        self.preprocessing_mode = str(config.get("preprocessing_mode", "raw_rgb"))
        self.transform = build_eval_transform(int(config.get("image_size", 224)))
        resolved_device = device_name
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved_device
        self.model = create_model(config["model"], num_classes=len(self.class_names)).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def score(self, image, cornea_mask_image) -> dict[str, float]:
        import torch  # type: ignore

        prepared = apply_variant(image, self.preprocessing_mode, cornea_mask_image)
        tensor = self.transform(prepared).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            logits_np = logits.detach().cpu().numpy()[0]
        payload: dict[str, float] = {}
        for index, class_name in enumerate(self.class_names):
            payload[f"pattern_prob_{class_name}"] = float(probabilities[index])
            payload[f"pattern_logit_{class_name}"] = float(logits_np[index])
        payload["pattern_pred_confidence"] = float(np.max(probabilities))
        payload["pattern_pred_index"] = float(np.argmax(probabilities))
        return payload


def row_features(row: pd.Series, repo_root: Path, pattern_scorer: PatternScorer | None) -> dict[str, object]:
    raw_image_path = resolve_repo_path(repo_root, row["raw_image_path"])
    cornea_mask_path = resolve_repo_path(repo_root, row["cornea_mask_path"])

    image = safe_open_image(raw_image_path).convert("RGB")
    cornea_mask_image = load_cornea_mask(cornea_mask_path)
    cornea_mask = normalize_mask_array(cornea_mask_image)
    circle = fit_cornea_circle(cornea_mask)
    rgb_array = np.asarray(image, dtype=np.float32)
    response, z_response = build_green_response(rgb_array, cornea_mask)
    proxy_masks = build_proxy_masks(z_response, cornea_mask)
    primary_mask = proxy_masks["primary"]

    distance_map = normalized_distance_map(image.height, image.width, circle.center_x, circle.center_y, circle.radius)
    lesion_distances = distance_map[primary_mask]
    cornea_area = float(max(cornea_mask.sum(), 1))
    positive_response = np.clip(z_response, 0.0, None)
    cornea_z_values = z_response[cornea_mask]
    lesion_z_values = z_response[primary_mask]

    central_zone = (distance_map <= ZONE_RADII["central"]) & cornea_mask
    paracentral_zone = (distance_map > ZONE_RADII["central"]) & (distance_map <= ZONE_RADII["paracentral"]) & cornea_mask
    peripheral_zone = (distance_map > ZONE_RADII["paracentral"]) & cornea_mask

    component_stats = component_features(primary_mask, distance_map)
    lesion_centroid_x = float(component_stats.pop("lesion_centroid_x", circle.center_x))
    lesion_centroid_y = float(component_stats.pop("lesion_centroid_y", circle.center_y))
    lesion_centroid_dist_norm = float(
        math.sqrt(((lesion_centroid_x - circle.center_x) ** 2) + ((lesion_centroid_y - circle.center_y) ** 2))
        / max(circle.radius, 1e-6)
    )

    central_weight = np.clip(1.0 - distance_map, 0.0, 1.0)
    central_response_weighted_area = float((positive_response * central_weight)[cornea_mask].sum() / cornea_area)
    response_area_ratio_t1_5_to_t0_5 = float(proxy_masks["t1_5"].sum() / max(float(proxy_masks["t0_5"].sum()), 1.0))
    if lesion_z_values.size:
        lesion_q75 = float(np.quantile(lesion_z_values, 0.75))
        lesion_q90 = float(np.quantile(lesion_z_values, 0.90))
        lesion_mass_total = float(np.clip(lesion_z_values, 0.0, None).sum())
        lesion_mass_top25_fraction = float(np.clip(lesion_z_values[lesion_z_values >= lesion_q75], 0.0, None).sum() / max(lesion_mass_total, 1e-6))
        lesion_mass_top10_fraction = float(np.clip(lesion_z_values[lesion_z_values >= lesion_q90], 0.0, None).sum() / max(lesion_mass_total, 1e-6))
    else:
        lesion_mass_top25_fraction = 0.0
        lesion_mass_top10_fraction = 0.0

    feature_row: dict[str, object] = {
        "image_id": str(row["image_id"]),
        "split": str(row["split"]),
        "severity_label": str(row["task_severity_5class"]),
        "pattern_label": str(row["task_pattern_3class"]),
        "tg_label": str(row["task_tg_5class"]),
        "raw_image_path": str(raw_image_path),
        "cornea_mask_path": str(cornea_mask_path),
        **circle.to_feature_dict(image.width, image.height),
        "green_response_mean": float(np.mean(response[cornea_mask])),
        "green_response_median": float(np.median(response[cornea_mask])),
        "green_response_std": float(np.std(response[cornea_mask])),
        "green_response_max": float(np.max(response[cornea_mask])),
        "green_response_top_decile": stable_quantile(response[cornea_mask], 0.9),
        "green_response_z_mean": float(np.mean(cornea_z_values)) if cornea_z_values.size else 0.0,
        "green_response_z_std": float(np.std(cornea_z_values)) if cornea_z_values.size else 0.0,
        "green_response_z_top_decile": stable_quantile(cornea_z_values, 0.9),
        "green_response_z_q95": stable_quantile(cornea_z_values, 0.95),
        "response_area_frac_t0_5": float(proxy_masks["t0_5"].sum() / cornea_area),
        "response_area_frac_t1_0": float(proxy_masks["t1_0"].sum() / cornea_area),
        "response_area_frac_t1_5": float(proxy_masks["t1_5"].sum() / cornea_area),
        "response_area_frac_t2_0": float(proxy_masks["t2_0"].sum() / cornea_area),
        "response_area_frac_t2_5": float(proxy_masks["t2_5"].sum() / cornea_area),
        "response_area_frac_t3_0": float(proxy_masks["t3_0"].sum() / cornea_area),
        "response_area_frac_t3_5": float(proxy_masks["t3_5"].sum() / cornea_area),
        "response_weighted_area": float(positive_response[cornea_mask].sum() / cornea_area),
        "response_weighted_area_t1_0": float(positive_response[proxy_masks["t1_0"]].sum() / cornea_area),
        "response_weighted_area_t1_5": float(positive_response[proxy_masks["t1_5"]].sum() / cornea_area),
        "central_response_weighted_area": central_response_weighted_area,
        "response_area_ratio_t1_5_to_t0_5": response_area_ratio_t1_5_to_t0_5,
        "lesion_mass_top25_fraction": lesion_mass_top25_fraction,
        "lesion_mass_top10_fraction": lesion_mass_top10_fraction,
        "response_peak_z": float(np.max(cornea_z_values)) if cornea_z_values.size else 0.0,
        "lesion_centroid_dist_norm": lesion_centroid_dist_norm,
        "lesion_min_dist_norm": float(np.min(lesion_distances)) if lesion_distances.size else 1.5,
        "lesion_fraction_in_central": float(primary_mask[central_zone].sum() / max(primary_mask.sum(), 1)),
        "lesion_fraction_in_paracentral": float(primary_mask[paracentral_zone].sum() / max(primary_mask.sum(), 1)),
        "lesion_fraction_in_peripheral": float(primary_mask[peripheral_zone].sum() / max(primary_mask.sum(), 1)),
        "central_minus_peripheral_fraction": float(
            (primary_mask[central_zone].sum() - primary_mask[peripheral_zone].sum()) / max(primary_mask.sum(), 1)
        ),
        "paracentral_minus_peripheral_fraction": float(
            (primary_mask[paracentral_zone].sum() - primary_mask[peripheral_zone].sum()) / max(primary_mask.sum(), 1)
        ),
        "central_zone_occupancy": float(primary_mask[central_zone].sum() / max(central_zone.sum(), 1)),
        "paracentral_zone_occupancy": float(primary_mask[paracentral_zone].sum() / max(paracentral_zone.sum(), 1)),
        "peripheral_zone_occupancy": float(primary_mask[peripheral_zone].sum() / max(peripheral_zone.sum(), 1)),
        "lesion_positive_pixels": int(primary_mask.sum()),
        "lesion_positive_mean_z": float(np.mean(lesion_z_values)) if lesion_z_values.size else 0.0,
        "lesion_positive_max_z": float(np.max(lesion_z_values)) if lesion_z_values.size else 0.0,
        **component_stats,
    }
    if pattern_scorer is not None:
        feature_row.update(pattern_scorer.score(image, cornea_mask_image))
        if "pattern_prob_flaky" in feature_row:
            feature_row["pattern_flaky_x_response_area_t1_0"] = float(feature_row["pattern_prob_flaky"]) * float(
                feature_row["response_area_frac_t1_0"]
            )
            feature_row["pattern_flaky_x_component_count"] = float(feature_row["pattern_prob_flaky"]) * float(
                feature_row["lesion_component_count"]
            )
        if "pattern_prob_point_flaky_mixed" in feature_row:
            feature_row["pattern_mixed_x_central_occupancy"] = float(feature_row["pattern_prob_point_flaky_mixed"]) * float(
                feature_row["central_zone_occupancy"]
            )
        if "pattern_pred_confidence" in feature_row:
            feature_row["pattern_confidence_x_response_weighted_area"] = float(feature_row["pattern_pred_confidence"]) * float(
                feature_row["response_weighted_area"]
            )
    return feature_row


def build_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    table = pd.DataFrame(rows)
    numeric_columns = [column for column in table.columns if pd.api.types.is_numeric_dtype(table[column])]
    summary = {
        "rows": int(len(table)),
        "columns": list(table.columns),
        "numeric_column_count": int(len(numeric_columns)),
        "splits": {str(key): int(value) for key, value in table["split"].value_counts().sort_index().items()},
        "severity_counts": {
            str(key): int(value) for key, value in table["severity_label"].value_counts().sort_index().items()
        },
    }
    write_json(output_path, summary)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    split_path = Path(args.split_file).resolve()
    output_path = Path(args.output_path).resolve()
    repo_root = infer_repo_root(manifest_path, args.repo_root)

    manifest_df = pd.read_csv(manifest_path)
    split_df = pd.read_csv(split_path)
    split_df["image_id"] = split_df["image_id"].astype(str)
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    table = manifest_df.merge(split_df[["image_id", "split"]], on="image_id", how="inner")
    table = table.sort_values(by=["split", "image_id"], kind="stable").reset_index(drop=True)

    pattern_scorer = None
    if args.pattern_config and args.pattern_checkpoint:
        pattern_scorer = PatternScorer(args.pattern_config, args.pattern_checkpoint, args.pattern_device)

    rows = [row_features(row, repo_root, pattern_scorer) for _, row in table.iterrows()]
    write_csv_rows(output_path, rows)
    build_summary(rows, output_path.parent / f"{output_path.stem}_summary.json")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
