from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import random

from config import EDAConfig, build_config, load_yaml_overrides, runtime_summary
from dataset_manifest import (
    audit_rows,
    build_manifest,
    export_manifest,
    folder_tree_rows,
    label_distribution_rows,
    manifest_overview_markdown,
)
from utils_duplicates import build_duplicate_rows
from utils_embeddings import extract_embeddings, project_embedding_table, summarize_projection_rows
from utils_image_stats import compute_image_stats, summarize_small_data_risk
from utils_io import ensure_directories, safe_open_image, setup_logging, write_csv_rows, write_json, write_text
from utils_masks import overlay_mask, summarize_masks
from utils_preprocessing import apply_variant, available_variants, sample_variant_images
from utils_viz import save_bar_chart, save_comparison_grid, save_histogram, save_montage, save_scatter_plot


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Paper-grounded EDA runner for the SUSTech-SYSU corneal ulcer dataset.")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--config", default="configs/eda.yaml")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-duplicates", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-mask-analysis", action="store_true")
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logging()

    config = build_config()
    config = load_yaml_overrides(config, args.config)
    config = apply_cli_overrides(config, args)
    random.seed(config.seed)

    ensure_directories(
        [
            config.manifest_dir,
            config.cleaned_metadata_dir,
            config.split_dir,
            config.figures_dir,
            config.tables_dir,
            config.cache_dir,
            config.embeddings_dir,
            config.reports_dir,
        ]
    )

    requested_device = args.device or "auto"
    runtime = runtime_summary(config, requested_device)
    write_json(config.reports_dir / "runtime_summary.json", runtime)

    manifest = build_manifest(config.data_root, logger)
    export_manifest(
        manifest,
        config.manifest_dir / "manifest.csv",
        config.manifest_dir / "manifest.parquet",
        logger,
    )
    write_csv_rows(config.tables_dir / "manifest.csv", manifest)

    audit = audit_rows(manifest, config.data_root)
    write_csv_rows(config.cleaned_metadata_dir / "audit_summary.csv", audit)
    write_csv_rows(config.tables_dir / "dataset_audit.csv", audit)

    folder_rows = folder_tree_rows(config.data_root)
    write_csv_rows(config.cleaned_metadata_dir / "folder_tree.csv", folder_rows)

    label_rows = label_distribution_rows(manifest)
    write_csv_rows(config.cleaned_metadata_dir / "label_summary.csv", label_rows)
    write_csv_rows(config.tables_dir / "label_distributions.csv", label_rows)

    for task_name, title in (("task_pattern_3class", "Pattern label distribution"),):
        rows = [row for row in label_rows if row["task"] == task_name]
        save_bar_chart(rows, "label", "count", title, config.figures_dir / f"{task_name}_distribution.png", logger)

    image_stats_rows: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    projection_summary_rows: list[dict[str, object]] = []
    preprocessing_outputs: list[dict[str, object]] = []

    image_stage_ready = bool(runtime.get("PIL", False))
    if image_stage_ready:
        logger.info("Computing image statistics for %d raw images", len(manifest))
        image_stats_rows = [compute_image_stats(Path(str(row["raw_image_path"])), logger) for row in manifest]
        save_image_statistics_figures(image_stats_rows, config, logger)
        save_random_and_class_montages(manifest, config, logger)
        save_extreme_example_panels(manifest, image_stats_rows, config, logger)
    else:
        logger.warning("Skipping image-dependent stages because Pillow is not installed.")
    write_csv_rows(config.tables_dir / "image_stats.csv", image_stats_rows)
    write_csv_rows(config.cleaned_metadata_dir / "image_stats.csv", image_stats_rows)

    if image_stage_ready and not args.skip_mask_analysis and config.masks_available:
        logger.info("Summarizing cornea and ulcer masks")
        mask_rows = summarize_masks(manifest, logger)
        save_mask_overlay_examples(manifest, config, logger)
    write_csv_rows(config.tables_dir / "mask_stats.csv", mask_rows)
    write_csv_rows(config.cleaned_metadata_dir / "mask_stats.csv", mask_rows)

    if image_stage_ready and not args.skip_preprocessing:
        logger.info("Generating preprocessing comparison figures")
        sample_rows = stratified_samples(manifest, per_label=2, seed=config.seed)
        for variant_name in available_variants():
            images = sample_variant_images(sample_rows, variant_name, logger)
            save_montage(images, config.figures_dir / f"preprocessing_{variant_name}.png", logger)
        preprocessing_outputs = build_preprocessing_comparison_rows(sample_rows, logger)
        save_comparison_grid(
            preprocessing_outputs,
            config.figures_dir / "preprocessing_comparison_grid.png",
            logger,
            tile_size=(180, 180),
        )

    if not args.skip_duplicates and image_stage_ready:
        logger.info("Running duplicate and near-duplicate checks")
        duplicate_rows = build_duplicate_rows(manifest, logger)
    write_csv_rows(config.tables_dir / "duplicate_candidates.csv", duplicate_rows)

    if not args.skip_embeddings:
        logger.info("Running ConvNeXtV2-focused embedding extraction")
        embedding_rows.extend(
            extract_embeddings(
                manifest,
                config.embedding_backbones,
                config.embeddings_dir,
                str(runtime["resolved_device"]),
                config.batch_size,
                logger,
                representation_name="raw_rgb",
            )
        )
        if image_stage_ready and config.embedding_compare_variant != "raw_rgb":
            embedding_rows.extend(
                extract_embeddings(
                    manifest,
                    config.embedding_backbones,
                    config.embeddings_dir,
                    str(runtime["resolved_device"]),
                    config.batch_size,
                    logger,
                    representation_name=config.embedding_compare_variant,
                )
            )
        projection_rows = project_embedding_table(embedding_rows, manifest, logger)
        projection_summary_rows = summarize_projection_rows(projection_rows)
        embedding_rows = merge_embedding_summaries(embedding_rows, projection_summary_rows)
        save_embedding_scatter_figures(projection_rows, config, logger)
        save_embedding_outlier_panels(manifest, projection_rows, config, logger)

    write_csv_rows(config.tables_dir / "embedding_summary.csv", embedding_rows)
    write_csv_rows(config.tables_dir / "embedding_projection_summary.csv", projection_rows)

    small_data_rows = summarize_small_data_risk(image_stats_rows, manifest)
    write_csv_rows(config.tables_dir / "small_data_risks.csv", small_data_rows)

    write_text(config.reports_dir / "dataset_audit_report.md", manifest_overview_markdown(manifest, audit))
    write_text(
        config.reports_dir / "leakage_report.md",
        compose_leakage_report(manifest, duplicate_rows, projection_summary_rows),
    )
    write_text(
        config.reports_dir / "split_recommendations.md",
        compose_split_recommendations(manifest, config),
    )
    write_text(
        config.reports_dir / "eda_summary.md",
        compose_eda_summary(
            manifest,
            audit,
            image_stats_rows,
            mask_rows,
            duplicate_rows,
            embedding_rows,
            projection_rows,
            runtime,
            config,
        ),
    )
    write_text(
        config.reports_dir / "model_readiness_summary.md",
        compose_model_readiness_summary(manifest, small_data_rows, runtime, config),
    )
    write_text(
        config.reports_dir / "image_stats_summary.md",
        compose_image_stats_summary(image_stats_rows, config),
    )
    write_text(
        config.reports_dir / "preprocessing_comparison_summary.md",
        compose_preprocessing_comparison_summary(config, preprocessing_outputs, image_stage_ready, args.skip_preprocessing),
    )
    write_text(
        config.reports_dir / "embedding_analysis_summary.md",
        compose_embedding_analysis_summary(embedding_rows, projection_rows, projection_summary_rows, config, args.skip_embeddings),
    )
    return 0


def apply_cli_overrides(config: EDAConfig, args) -> EDAConfig:
    if args.data_root is not None:
        config.data_root = Path(args.data_root)
    if args.output_root is not None:
        config.output_root = Path(args.output_root)
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.seed is not None:
        config.seed = args.seed
    return config


def stratified_samples(manifest: list[dict[str, object]], per_label: int = 2, seed: int = 42) -> list[dict[str, object]]:
    buckets: dict[str, list[dict[str, object]]] = {}
    rng = random.Random(seed)
    for row in manifest:
        buckets.setdefault(str(row["task_pattern_3class"]), []).append(row)
    sampled: list[dict[str, object]] = []
    for rows in buckets.values():
        rows_copy = list(rows)
        rng.shuffle(rows_copy)
        sampled.extend(rows_copy[:per_label])
    return sampled


def save_image_statistics_figures(image_stats_rows: list[dict[str, object]], config: EDAConfig, logger) -> None:
    numeric_metrics = {
        "width": "Image width distribution",
        "height": "Image height distribution",
        "aspect_ratio": "Aspect ratio distribution",
        "filesize_bytes": "File size distribution",
        "brightness": "Brightness distribution",
        "contrast": "Contrast distribution",
        "blur_proxy": "Blur proxy distribution",
        "entropy": "Entropy distribution",
        "green_dominance": "Green dominance distribution",
    }
    readable_rows = [row for row in image_stats_rows if row.get("readable")]
    for key, title in numeric_metrics.items():
        values = [float(row[key]) for row in readable_rows if key in row]
        save_histogram(values, title, config.figures_dir / f"{key}_distribution.png", logger)


def save_random_and_class_montages(manifest: list[dict[str, object]], config: EDAConfig, logger) -> None:
    rng = random.Random(config.seed)
    sample_size = min(config.montage_sample_count, len(manifest))
    random_rows = rng.sample(manifest, sample_size) if sample_size else []
    save_montage(load_raw_images(random_rows, "image_id", logger), config.figures_dir / "random_raw_samples.png", logger)

    class_buckets: dict[str, list[dict[str, object]]] = {}
    for row in manifest:
        class_buckets.setdefault(str(row["task_pattern_3class"]), []).append(row)
    for label, rows in sorted(class_buckets.items()):
        subset = rows[: config.extreme_sample_count]
        save_montage(
            load_raw_images(subset, "task_pattern_3class", logger),
            config.figures_dir / f"class_montage_{label}.png",
            logger,
        )


def save_extreme_example_panels(
    manifest: list[dict[str, object]],
    image_stats_rows: list[dict[str, object]],
    config: EDAConfig,
    logger,
) -> None:
    if not image_stats_rows:
        return
    manifest_map = {str(row["image_id"]): row for row in manifest}
    readable_rows = [row for row in image_stats_rows if row.get("readable")]
    if not readable_rows:
        return

    for metric, orderings in {
        "brightness": ("low", "high"),
        "blur_proxy": ("low", "high"),
        "green_dominance": ("low", "high"),
    }.items():
        sorted_rows = sorted(readable_rows, key=lambda row: float(row[metric]))
        low_ids = [str(row["image_id"]) for row in sorted_rows[: config.extreme_sample_count]]
        high_ids = [str(row["image_id"]) for row in sorted_rows[-config.extreme_sample_count :]]
        for suffix, image_ids in zip(orderings, (low_ids, high_ids), strict=True):
            selected_manifest_rows = [manifest_map[image_id] for image_id in image_ids if image_id in manifest_map]
            save_montage(
                load_raw_images(selected_manifest_rows, "image_id", logger),
                config.figures_dir / f"extreme_{metric}_{suffix}.png",
                logger,
            )


def save_mask_overlay_examples(manifest: list[dict[str, object]], config: EDAConfig, logger) -> None:
    images: list[tuple[str, object]] = []
    selected = [row for row in manifest if row.get("has_cornea_mask")][: config.extreme_sample_count]
    for row in selected:
        try:
            base_image = safe_open_image(Path(str(row["raw_image_path"])))
            cornea_mask = safe_open_image(Path(str(row["cornea_mask_path"])))
            composite = overlay_mask(base_image, cornea_mask, color=(0, 255, 0), alpha=90)
            if row.get("has_ulcer_mask"):
                ulcer_mask = safe_open_image(Path(str(row["ulcer_mask_path"])))
                composite = overlay_mask(composite, ulcer_mask, color=(255, 0, 0), alpha=120)
            images.append((str(row["image_id"]), composite))
        except Exception as exc:
            logger.warning("Skipping mask overlay montage for %s: %s", row["image_id"], exc)
    save_montage(images, config.figures_dir / "mask_overlay_examples.png", logger)


def build_preprocessing_comparison_rows(sample_rows: list[dict[str, object]], logger) -> list[dict[str, object]]:
    comparison_variants = [
        "raw_rgb",
        "cornea_crop_scale_v1",
        "crop_scale_raw_multiscale",
        "blue_channel_removed",
        "grayscale",
        "gaussian_blur",
        "otsu_threshold",
        "masked_highlight_proxy",
    ]
    rows: list[dict[str, object]] = []
    for row in sample_rows:
        try:
            image = safe_open_image(Path(str(row["raw_image_path"])))
        except Exception as exc:
            logger.warning("Skipping preprocessing comparison row %s: %s", row["image_id"], exc)
            continue
        cornea_mask = None
        cornea_mask_path = str(row.get("cornea_mask_path") or "")
        if cornea_mask_path:
            try:
                cornea_mask = safe_open_image(Path(cornea_mask_path))
            except Exception:
                cornea_mask = None
        comparison_row: dict[str, object] = {"image_id": row["image_id"]}
        for variant_name in comparison_variants:
            comparison_row[variant_name] = apply_variant(image, variant_name, cornea_mask)
        rows.append(comparison_row)
    return rows


def save_embedding_scatter_figures(projection_rows: list[dict[str, object]], config: EDAConfig, logger) -> None:
    if not projection_rows:
        return
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in projection_rows:
        grouped.setdefault((str(row["representation"]), str(row["backbone"])), []).append(row)

    for (representation, backbone), rows in grouped.items():
        for color_key in ("task_pattern_3class",):
            save_scatter_plot(
                rows,
                "proj_x",
                "proj_y",
                color_key,
                f"{backbone} {representation} colored by {color_key}",
                config.figures_dir / f"embedding_{backbone}_{representation}_{color_key}.png",
                logger,
            )


def save_embedding_outlier_panels(
    manifest: list[dict[str, object]],
    projection_rows: list[dict[str, object]],
    config: EDAConfig,
    logger,
) -> None:
    if not projection_rows:
        return
    manifest_map = {str(row["image_id"]): row for row in manifest}
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in projection_rows:
        grouped.setdefault((str(row["representation"]), str(row["backbone"])), []).append(row)
    for (representation, backbone), rows in grouped.items():
        top_rows = sorted(rows, key=lambda row: float(row["outlier_zscore"]), reverse=True)[: config.extreme_sample_count]
        selected_manifest = [manifest_map[str(row["image_id"])] for row in top_rows if str(row["image_id"]) in manifest_map]
        save_montage(
            load_raw_images(selected_manifest, "image_id", logger),
            config.figures_dir / f"embedding_outliers_{backbone}_{representation}.png",
            logger,
        )


def load_raw_images(rows: list[dict[str, object]], label_key: str, logger) -> list[tuple[str, object]]:
    images: list[tuple[str, object]] = []
    for row in rows:
        try:
            image = safe_open_image(Path(str(row["raw_image_path"])))
            images.append((str(row[label_key]), image))
        except Exception as exc:
            logger.warning("Skipping montage image %s: %s", row.get("image_id"), exc)
    return images


def merge_embedding_summaries(
    embedding_rows: list[dict[str, object]],
    projection_summary_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    projection_map = {
        (str(row["representation"]), str(row["backbone"])): row for row in projection_summary_rows
    }
    merged: list[dict[str, object]] = []
    for row in embedding_rows:
        key = (str(row["representation"]), str(row["backbone"]))
        projection_row = projection_map.get(key, {})
        merged.append({**row, **projection_row})
    return merged


def compose_leakage_report(
    manifest: list[dict[str, object]],
    duplicate_rows: list[dict[str, object]],
    projection_summary_rows: list[dict[str, object]],
) -> str:
    ulcer_mask_rows = sum(1 for row in manifest if row["has_ulcer_mask"])
    cross_label_duplicates = sum(1 for row in duplicate_rows if row.get("cross_label_suspicion"))
    lines = [
        "# Leakage Report",
        "",
        "- No official split files were provided by the dataset.",
        "- No patient or visit identifiers are present, so patient-level leakage cannot be ruled out.",
        "- Augmentation must only happen after split assignment.",
        "- Cornea and ulcer masks/overlays are derived assets and must stay with their parent image ID.",
        f"- Ulcer masks exist for {ulcer_mask_rows} of {len(manifest)} images and are label-correlated, so mask presence itself must not become a proxy feature.",
        f"- Duplicate and near-duplicate checks found {len(duplicate_rows)} candidate rows, including {cross_label_duplicates} cross-label suspicions.",
    ]
    if projection_summary_rows:
        lines.append(
            f"- Embedding-neighbor mismatch ratios are summarized for {len(projection_summary_rows)} representation/backbone combinations; use them before locking split files."
        )
    return "\n".join(lines)


def compose_split_recommendations(manifest: list[dict[str, object]], config: EDAConfig) -> str:
    return "\n".join(
        [
            "# Split Recommendations",
            "",
            "## Defaults",
            "",
            "- Build splits at the raw-image ID level.",
            "- Keep all masks and overlays with the same split as the parent image.",
            "- Prefer duplicate-aware repeated stratified splits or grouped cross-validation.",
            "- Do not finalize patient-aware claims because patient IDs are absent.",
            "- Apply augmentation only after split assignment.",
            "",
            "## Stage 3 alignment",
            "",
            f"- Use raw RGB + {config.embedding_backbones[0]} as the primary classification-aligned baseline path.",
            f"- Use {config.embedding_compare_variant} as the matched paper-inspired comparison path.",
            "",
            "## Higher-risk tasks",
            "",
            "- Any segmentation-assisted classification should be treated as a subset experiment because ulcer masks exist on only part of the cohort.",
        ]
    )


def compose_eda_summary(
    manifest: list[dict[str, object]],
    audit: list[dict[str, object]],
    image_stats_rows: list[dict[str, object]],
    mask_rows: list[dict[str, object]],
    duplicate_rows: list[dict[str, object]],
    embedding_rows: list[dict[str, object]],
    projection_rows: list[dict[str, object]],
    runtime: dict[str, object],
    config: EDAConfig,
) -> str:
    audit_map = {row["metric"]: row["value"] for row in audit}
    readable_rows = [row for row in image_stats_rows if row.get("readable")]
    mean_brightness = mean_metric(readable_rows, "brightness")
    mean_contrast = mean_metric(readable_rows, "contrast")
    return "\n".join(
        [
            "# EDA Summary",
            "",
            "## Runtime",
            "",
            f"- Resolved device: {runtime['resolved_device']}",
            f"- Torch available: {runtime['torch']}",
            f"- Pillow available: {runtime['PIL']}",
            f"- Embedding backbones: {', '.join(config.embedding_backbones)}",
            f"- Embedding comparison variant: {config.embedding_compare_variant}",
            "",
            "## Dataset recap",
            "",
            f"- Manifest rows: {len(manifest)}",
            f"- Raw images: {audit_map.get('raw_images', 0)}",
            f"- Cornea masks: {audit_map.get('cornea_masks', 0)}",
            f"- Ulcer masks: {audit_map.get('ulcer_masks', 0)}",
            "",
            "## Computed outputs",
            "",
            f"- Image stats rows: {len(image_stats_rows)}",
            f"- Mean brightness: {mean_brightness}",
            f"- Mean contrast: {mean_contrast}",
            f"- Mask stats rows: {len(mask_rows)}",
            f"- Duplicate candidate rows: {len(duplicate_rows)}",
            f"- Embedding summary rows: {len(embedding_rows)}",
            f"- Embedding projection rows: {len(projection_rows)}",
            "",
            "## Interpretation",
            "",
            "- This EDA is a classification-readiness pass, not a model benchmark.",
            "- Paper claims are not project results; only on-disk counts and computed artifacts are treated as project findings.",
            "- Raw RGB and one Diagnostics-inspired preprocessing proxy are both preserved for Stage 3 comparison.",
        ]
    )


def compose_model_readiness_summary(
    manifest: list[dict[str, object]],
    small_data_rows: list[dict[str, object]],
    runtime: dict[str, object],
    config: EDAConfig,
) -> str:
    risk_map = {row["risk"]: row["value"] for row in small_data_rows}
    return "\n".join(
        [
            "# Model Readiness Summary",
            "",
            f"- Dataset rows available for baseline classification: {len(manifest)}",
            f"- Resolved device for feature extraction/training: {runtime['resolved_device']}",
            f"- Pattern imbalance ratio: {risk_map.get('pattern_imbalance_ratio', 'n/a')}",
            f"- Ulcer-mask subset ratio: {risk_map.get('ulcer_mask_subset_ratio', 'n/a')}",
            "",
            "## Recommendation",
            "",
            f"- Start Stage 3 with raw RGB + {config.embedding_backbones[0]} on the confirmed 3-class pattern task.",
            f"- Run {config.embedding_compare_variant} as the matched paper-inspired baseline rather than assuming preprocessing is superior.",
            "- Keep imbalance-aware metrics and duplicate-aware split validation mandatory.",
        ]
    )


def compose_image_stats_summary(image_stats_rows: list[dict[str, object]], config: EDAConfig) -> str:
    readable_rows = [row for row in image_stats_rows if row.get("readable")]
    if not readable_rows:
        return "# Image Stats Summary\n\n- Image statistics were not computed in this run."

    low_brightness = sum(1 for row in readable_rows if float(row["brightness"]) < config.brightness_low_threshold)
    high_brightness = sum(1 for row in readable_rows if float(row["brightness"]) > config.brightness_high_threshold)
    low_blur = sum(1 for row in readable_rows if float(row["blur_proxy"]) < config.blur_low_threshold)
    return "\n".join(
        [
            "# Image Stats Summary",
            "",
            f"- Readable images analyzed: {len(readable_rows)}",
            f"- Mean width: {mean_metric(readable_rows, 'width')}",
            f"- Mean height: {mean_metric(readable_rows, 'height')}",
            f"- Mean file size bytes: {mean_metric(readable_rows, 'filesize_bytes')}",
            f"- Mean brightness: {mean_metric(readable_rows, 'brightness')}",
            f"- Mean contrast: {mean_metric(readable_rows, 'contrast')}",
            f"- Mean blur proxy: {mean_metric(readable_rows, 'blur_proxy')}",
            f"- Mean entropy: {mean_metric(readable_rows, 'entropy')}",
            f"- Mean green dominance: {mean_metric(readable_rows, 'green_dominance')}",
            f"- Low-brightness images below threshold {config.brightness_low_threshold}: {low_brightness}",
            f"- High-brightness images above threshold {config.brightness_high_threshold}: {high_brightness}",
            f"- Low-blur-proxy images below threshold {config.blur_low_threshold}: {low_blur}",
        ]
    )


def compose_preprocessing_comparison_summary(
    config: EDAConfig,
    preprocessing_outputs: list[dict[str, object]],
    image_stage_ready: bool,
    skip_preprocessing: bool,
) -> str:
    if not image_stage_ready:
        return "# Preprocessing Comparison Summary\n\n- Preprocessing comparison was unavailable because Pillow was not installed."
    if skip_preprocessing:
        return "# Preprocessing Comparison Summary\n\n- Preprocessing comparison was skipped for this run."
    return "\n".join(
        [
            "# Preprocessing Comparison Summary",
            "",
            "- The comparison is Diagnostics-inspired, not a faithful reproduction of every paper step.",
            f"- Selected embedding comparison variant: {config.embedding_compare_variant}",
            f"- Variants visualized: {', '.join(available_variants())}",
            f"- Comparison grid rows rendered: {len(preprocessing_outputs)}",
            "- Raw RGB remains the baseline representation; the preprocessing path is treated as a controlled comparison rather than an assumed improvement.",
        ]
    )


def compose_embedding_analysis_summary(
    embedding_rows: list[dict[str, object]],
    projection_rows: list[dict[str, object]],
    projection_summary_rows: list[dict[str, object]],
    config: EDAConfig,
    skip_embeddings: bool,
) -> str:
    if skip_embeddings:
        return "# Embedding Analysis Summary\n\n- Embedding analysis was skipped for this run."
    if not embedding_rows:
        return "# Embedding Analysis Summary\n\n- Embedding analysis did not produce artifacts."

    lines = [
        "# Embedding Analysis Summary",
        "",
        f"- Backbone focus: {', '.join(config.embedding_backbones)}",
        f"- Representations compared: raw_rgb, {config.embedding_compare_variant}",
        f"- Embedding artifact rows: {len(embedding_rows)}",
        f"- Projection rows: {len(projection_rows)}",
        "",
        "## Representation summaries",
        "",
    ]
    for row in projection_summary_rows:
        lines.extend(
            [
                f"- {row['backbone']} / {row['representation']}: method={row['projection_method']}, "
                f"pattern_mismatch={row['pattern_neighbor_mismatch_ratio']}, "
                f"mean_neighbor_distance={row['mean_neighbor_cosine_distance']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Neighbor mismatch ratios are used here as EDA signals for label mixing, not as model-quality metrics.",
            "- Any apparent separation between raw and preprocessed embeddings should be treated as a hypothesis for Stage 3, not evidence that one path is inherently superior.",
        ]
    )
    return "\n".join(lines)


def mean_metric(rows: list[dict[str, object]], key: str) -> str:
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return "n/a"
    return f"{sum(values) / len(values):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
