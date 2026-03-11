from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import math

from utils_io import safe_open_image
from utils_preprocessing import apply_variant


def embedding_dependencies_ready() -> bool:
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


def _sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _load_backbone(name: str, logger: logging.Logger):
    import torch  # type: ignore

    if name == "resnet18":
        try:
            from torchvision.models import ResNet18_Weights, resnet18  # type: ignore

            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            model.fc = torch.nn.Identity()
            return model.eval()
        except Exception as exc:
            logger.warning("Failed to load torchvision resnet18: %s", exc)
            return None

    try:
        import timm  # type: ignore

        model = timm.create_model(name, pretrained=True, num_classes=0)
        return model.eval()
    except Exception as exc:
        logger.warning("Failed to load backbone %s: %s", name, exc)
        return None


def _image_to_tensor(image):
    import torch  # type: ignore

    try:
        from torchvision import transforms  # type: ignore

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        return transform(image)
    except Exception:
        import numpy as np  # type: ignore

        resized = image.convert("RGB").resize((224, 224))
        array = np.asarray(resized, dtype="float32") / 255.0
        array = np.transpose(array, (2, 0, 1))
        tensor = torch.tensor(array)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std


def _load_representation_image(row: dict[str, Any], representation_name: str, logger: logging.Logger):
    image = safe_open_image(Path(str(row["raw_image_path"])))
    if representation_name == "raw_rgb":
        return image.convert("RGB")

    cornea_mask = None
    cornea_mask_path = str(row.get("cornea_mask_path") or "")
    if cornea_mask_path:
        try:
            cornea_mask = safe_open_image(Path(cornea_mask_path))
        except Exception as exc:
            logger.warning("Could not load cornea mask for %s: %s", row["image_id"], exc)
    return apply_variant(image, representation_name, cornea_mask)


def extract_embeddings(
    manifest_rows: list[dict[str, Any]],
    backbones: list[str],
    output_dir: Path,
    device: str,
    batch_size: int,
    logger: logging.Logger,
    representation_name: str = "raw_rgb",
) -> list[dict[str, Any]]:
    if not embedding_dependencies_ready():
        logger.warning("Skipping embeddings because torch/numpy/sklearn are not available.")
        return []

    import numpy as np  # type: ignore
    import torch  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    rows: list[dict[str, Any]] = []
    variant_slug = _sanitize_name(representation_name)

    for backbone_name in backbones:
        logger.info("Extracting %s embeddings for representation %s", backbone_name, representation_name)
        model = _load_backbone(backbone_name, logger)
        if model is None:
            continue
        model = model.to(device_name)
        embeddings: list[np.ndarray] = []
        image_ids: list[str] = []

        batch_tensors = []
        batch_ids: list[str] = []
        for row in manifest_rows:
            raw_path = Path(str(row["raw_image_path"]))
            if not raw_path.exists():
                continue
            try:
                image = _load_representation_image(row, representation_name, logger)
                batch_tensors.append(_image_to_tensor(image))
                batch_ids.append(str(row["image_id"]))
            except Exception as exc:
                logger.warning("Skipping embedding image %s for %s: %s", row["image_id"], representation_name, exc)
                continue
            if len(batch_tensors) >= batch_size:
                batch_output = _forward_batch(model, batch_tensors, device_name)
                embeddings.extend(batch_output)
                image_ids.extend(batch_ids)
                batch_tensors = []
                batch_ids = []
        if batch_tensors:
            batch_output = _forward_batch(model, batch_tensors, device_name)
            embeddings.extend(batch_output)
            image_ids.extend(batch_ids)

        if not embeddings:
            continue

        array = np.vstack(embeddings)
        artifact_path = output_dir / f"{variant_slug}__{backbone_name}_embeddings.npz"
        np.savez(artifact_path, image_ids=np.array(image_ids), embeddings=array)
        rows.append(
            {
                "representation": representation_name,
                "backbone": backbone_name,
                "embedding_rows": int(array.shape[0]),
                "embedding_dim": int(array.shape[1]),
                "device": device_name,
                "artifact_path": str(artifact_path),
            }
        )
    return rows


def _forward_batch(model, batch_tensors, device_name: str):
    import numpy as np  # type: ignore
    import torch  # type: ignore

    batch = torch.stack(batch_tensors).to(device_name)
    with torch.inference_mode():
        output = model(batch)
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
    return [np.atleast_2d(vector).reshape(1, -1) for vector in output]


def _project_embeddings(array, logger: logging.Logger):
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(array)
        return coords, "umap"
    except Exception as exc:
        logger.warning("UMAP projection unavailable, falling back to t-SNE: %s", exc)

    try:
        from sklearn.manifold import TSNE  # type: ignore

        perplexity = max(5, min(30, max(5, array.shape[0] // 20)))
        coords = TSNE(n_components=2, random_state=42, init="pca", perplexity=perplexity).fit_transform(array)
        return coords, "tsne"
    except Exception as exc:
        logger.warning("t-SNE projection unavailable: %s", exc)
        return None, "unavailable"


def _cosine_neighbor_metrics(array):
    import numpy as np  # type: ignore

    if len(array) < 2:
        return [""], [0.0]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = array / np.clip(norms, 1e-8, None)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    neighbor_indices = similarity.argmax(axis=1)
    neighbor_scores = similarity[np.arange(len(array)), neighbor_indices]
    neighbor_distances = 1.0 - neighbor_scores
    return neighbor_indices.tolist(), neighbor_distances.tolist()


def project_embedding_table(
    summary_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    if not summary_rows:
        return []

    import numpy as np  # type: ignore

    manifest_map = {str(row["image_id"]): row for row in manifest_rows}
    projection_rows: list[dict[str, Any]] = []

    for row in summary_rows:
        artifact_path = Path(str(row["artifact_path"]))
        if not artifact_path.exists():
            continue
        payload = np.load(artifact_path, allow_pickle=False)
        image_ids = [str(value) for value in payload["image_ids"].tolist()]
        array = payload["embeddings"]
        coords, method = _project_embeddings(array, logger)
        if coords is None:
            coords = np.zeros((len(image_ids), 2), dtype="float32")
        centroid = array.mean(axis=0, keepdims=True)
        outlier_scores = np.linalg.norm(array - centroid, axis=1)
        if len(outlier_scores) > 1:
            outlier_mean = float(outlier_scores.mean())
            outlier_std = float(outlier_scores.std()) or 1.0
            standardized = (outlier_scores - outlier_mean) / outlier_std
        else:
            standardized = np.zeros_like(outlier_scores)
        neighbor_indices, neighbor_distances = _cosine_neighbor_metrics(array)

        for index, image_id in enumerate(image_ids):
            manifest_row = manifest_map.get(image_id, {})
            neighbor_image_id = ""
            neighbor_pattern_match = ""
            neighbor_severity_match = ""
            neighbor_tg_match = ""
            if isinstance(neighbor_indices[index], int):
                neighbor_image_id = image_ids[neighbor_indices[index]]
                neighbor_row = manifest_map.get(neighbor_image_id, {})
                neighbor_pattern_match = bool(
                    manifest_row.get("task_pattern_3class") == neighbor_row.get("task_pattern_3class")
                )
                neighbor_severity_match = bool(
                    manifest_row.get("task_severity_5class") == neighbor_row.get("task_severity_5class")
                )
                neighbor_tg_match = bool(manifest_row.get("task_tg_5class") == neighbor_row.get("task_tg_5class"))

            projection_rows.append(
                {
                    "representation": row["representation"],
                    "backbone": row["backbone"],
                    "projection_method": method,
                    "image_id": image_id,
                    "proj_x": round(float(coords[index, 0]), 6),
                    "proj_y": round(float(coords[index, 1]), 6),
                    "outlier_score": round(float(outlier_scores[index]), 6),
                    "outlier_zscore": round(float(standardized[index]), 6),
                    "neighbor_image_id": neighbor_image_id,
                    "neighbor_cosine_distance": round(float(neighbor_distances[index]), 6),
                    "neighbor_pattern_match": neighbor_pattern_match,
                    "neighbor_severity_match": neighbor_severity_match,
                    "neighbor_tg_match": neighbor_tg_match,
                    "task_pattern_3class": manifest_row.get("task_pattern_3class", ""),
                    "task_severity_5class": manifest_row.get("task_severity_5class", ""),
                    "task_tg_5class": manifest_row.get("task_tg_5class", ""),
                }
            )
    logger.info("Projected embeddings for %d backbone/representation combinations", len(summary_rows))
    return projection_rows


def summarize_projection_rows(projection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in projection_rows:
        key = (str(row["representation"]), str(row["backbone"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (representation, backbone), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "representation": representation,
                "backbone": backbone,
                "projection_method": rows[0]["projection_method"],
                "rows": len(rows),
                "mean_neighbor_cosine_distance": round(
                    sum(float(row["neighbor_cosine_distance"]) for row in rows) / max(1, len(rows)),
                    6,
                ),
                "mean_outlier_zscore": round(
                    sum(abs(float(row["outlier_zscore"])) for row in rows) / max(1, len(rows)),
                    6,
                ),
                "pattern_neighbor_mismatch_ratio": round(
                    _mismatch_ratio(rows, "neighbor_pattern_match"),
                    6,
                ),
                "severity_neighbor_mismatch_ratio": round(
                    _mismatch_ratio(rows, "neighbor_severity_match"),
                    6,
                ),
                "tg_neighbor_mismatch_ratio": round(
                    _mismatch_ratio(rows, "neighbor_tg_match"),
                    6,
                ),
            }
        )
    return summary_rows


def _mismatch_ratio(rows: list[dict[str, Any]], key: str) -> float:
    values = [row[key] for row in rows if isinstance(row[key], bool)]
    if not values:
        return 0.0
    mismatches = sum(1 for value in values if not value)
    return mismatches / len(values)
