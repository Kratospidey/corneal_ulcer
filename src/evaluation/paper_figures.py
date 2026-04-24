from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import json

from config_utils import resolve_config, write_text
from data.label_utils import get_task_definition
from evaluation.metrics import compute_classification_metrics
from experiment_utils import build_experiment_name, resolve_device, setup_logging
from explainability.gradcam_utils import GradCAM, disable_inplace_relu, overlay_cam_on_image
from inference.inference_utils import load_image_for_inference
from model_factory import create_model, get_gradcam_target_layer
from utils_io import safe_open_image


FIGURE_FILENAMES = {
    "confusion": "01_test_confusion_matrix",
    "roc": "02_test_roc_curves",
    "pr": "03_test_pr_curves",
    "reliability": "04_test_reliability",
    "confidence": "05_test_confidence_histogram",
    "per_class": "06_test_per_class_metrics",
    "split_overview": "07_split_overview",
    "xai": "08_test_gradcam_gallery",
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate publication-ready paper figures for a trained pattern model.")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions-csv")
    parser.add_argument("--metrics-json")
    parser.add_argument("--output-root", default="outputs/paper_figures")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--xai-count", type=int, default=6)
    return parser


def _load_predictions(predictions_csv: Path) -> list[dict[str, Any]]:
    import csv

    with predictions_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def build_reliability_bins(probabilities: list[list[float]], y_true: list[int], num_bins: int = 10) -> list[dict[str, float]]:
    import numpy as np  # type: ignore

    probs = np.asarray(probabilities, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=int)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == y_true_arr).astype(float)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    rows: list[dict[str, float]] = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences >= lower) & (confidences < upper if upper < 1.0 else confidences <= upper)
        count = int(mask.sum())
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": float(count),
                "accuracy": float(correctness[mask].mean()) if count else 0.0,
                "confidence": float(confidences[mask].mean()) if count else 0.0,
            }
        )
    return rows


def select_xai_rows(prediction_rows: list[dict[str, Any]], class_names: list[str], xai_count: int) -> list[dict[str, Any]]:
    correct_rows = [row for row in prediction_rows if _to_bool(row.get("correct"))]
    incorrect_rows = [row for row in prediction_rows if not _to_bool(row.get("correct"))]

    def _sorted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda row: float(row.get("confidence", 0.0)), reverse=True)

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for class_name in class_names:
        for row in _sorted([row for row in correct_rows if str(row.get("true_label")) == class_name]):
            image_id = str(row.get("image_id"))
            if image_id in used_ids:
                continue
            selected.append(row)
            used_ids.add(image_id)
            break

    for class_name in class_names:
        for row in _sorted([row for row in incorrect_rows if str(row.get("true_label")) == class_name]):
            image_id = str(row.get("image_id"))
            if image_id in used_ids:
                continue
            selected.append(row)
            used_ids.add(image_id)
            if len(selected) >= xai_count:
                return selected[:xai_count]
            break

    for row in _sorted(incorrect_rows) + _sorted(correct_rows):
        image_id = str(row.get("image_id"))
        if image_id in used_ids:
            continue
        selected.append(row)
        used_ids.add(image_id)
        if len(selected) >= xai_count:
            break
    return selected[:xai_count]


def _configure_publication_style() -> list[str]:
    import matplotlib.pyplot as plt  # type: ignore

    plt.style.use("seaborn-v0_8-whitegrid")
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return palette


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def _plot_confusion_matrix(matrix: list[list[int]], class_names: list[str], output_dir: Path, palette: list[str]) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    counts = np.asarray(matrix, dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test Confusion Matrix")
    for row in range(counts.shape[0]):
        for col in range(counts.shape[1]):
            ax.text(
                col,
                row,
                f"{int(counts[row, col])}\n{normalized[row, col]:.2f}",
                ha="center",
                va="center",
                color="white" if normalized[row, col] > 0.55 else "black",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion")
    _save_figure(fig, output_dir, FIGURE_FILENAMES["confusion"])
    plt.close(fig)


def _plot_roc_pr_curves(
    y_true: list[int],
    probabilities: list[list[float]],
    class_names: list[str],
    output_dir: Path,
    palette: list[str],
) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # type: ignore
    from sklearn.preprocessing import label_binarize  # type: ignore

    probs = np.asarray(probabilities, dtype=float)
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    fig_roc, ax_roc = plt.subplots(figsize=(5.4, 4.6))
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
    for idx, class_name in enumerate(class_names):
        y_true_class = y_true_bin[:, idx]
        fpr, tpr, _ = roc_curve(y_true_class, probs[:, idx])
        auc = roc_auc_score(y_true_class, probs[:, idx])
        ax_roc.plot(fpr, tpr, linewidth=2, color=palette[idx % len(palette)], label=f"{class_name} (AUC={auc:.3f})")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("One-vs-Rest ROC Curves")
    ax_roc.legend(frameon=False, loc="lower right")
    _save_figure(fig_roc, output_dir, FIGURE_FILENAMES["roc"])
    plt.close(fig_roc)

    fig_pr, ax_pr = plt.subplots(figsize=(5.4, 4.6))
    for idx, class_name in enumerate(class_names):
        y_true_class = y_true_bin[:, idx]
        precision, recall, _ = precision_recall_curve(y_true_class, probs[:, idx])
        ap = average_precision_score(y_true_class, probs[:, idx])
        ax_pr.plot(recall, precision, linewidth=2, color=palette[idx % len(palette)], label=f"{class_name} (AP={ap:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("One-vs-Rest Precision-Recall Curves")
    ax_pr.legend(frameon=False, loc="lower left")
    _save_figure(fig_pr, output_dir, FIGURE_FILENAMES["pr"])
    plt.close(fig_pr)


def _plot_reliability(reliability_rows: list[dict[str, float]], ece: float, output_dir: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    centers = np.asarray([(row["lower"] + row["upper"]) / 2.0 for row in reliability_rows], dtype=float)
    accuracies = np.asarray([row["accuracy"] for row in reliability_rows], dtype=float)
    confidences = np.asarray([row["confidence"] for row in reliability_rows], dtype=float)
    widths = np.asarray([row["upper"] - row["lower"] for row in reliability_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.bar(centers, accuracies, width=widths * 0.95, alpha=0.65, color="#0072B2", label="Accuracy")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=1, label="Perfect calibration")
    ax.plot(centers, confidences, color="#D55E00", marker="o", linewidth=2, label="Confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Accuracy / confidence")
    ax.set_title(f"Reliability Diagram (ECE={ece:.3f})")
    ax.legend(frameon=False, loc="upper left")
    _save_figure(fig, output_dir, FIGURE_FILENAMES["reliability"])
    plt.close(fig)


def _plot_confidence_histogram(confidences: list[float], correctness: list[bool], output_dir: Path, palette: list[str]) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    correct_values = np.asarray([value for value, is_correct in zip(confidences, correctness, strict=True) if is_correct], dtype=float)
    incorrect_values = np.asarray([value for value, is_correct in zip(confidences, correctness, strict=True) if not is_correct], dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    bins = np.linspace(0.0, 1.0, 15)
    ax.hist(correct_values, bins=bins, alpha=0.7, label="Correct", color=palette[2], density=False)
    ax.hist(incorrect_values, bins=bins, alpha=0.7, label="Incorrect", color=palette[1], density=False)
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Number of samples")
    ax.set_title("Confidence Distribution on Test Set")
    ax.legend(frameon=False)
    _save_figure(fig, output_dir, FIGURE_FILENAMES["confidence"])
    plt.close(fig)


def _plot_per_class_metrics(classification_report: dict[str, Any], class_names: list[str], output_dir: Path, palette: list[str]) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for idx, metric_name in enumerate(metrics):
        values = [float(classification_report[class_name][metric_name]) for class_name in class_names]
        ax.bar(x + ((idx - 1) * width), values, width=width, label=metric_name.replace("-score", "").upper(), color=palette[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Test Metrics")
    ax.legend(frameon=False)
    _save_figure(fig, output_dir, FIGURE_FILENAMES["per_class"])
    plt.close(fig)


def _plot_split_overview(split_metrics: dict[str, dict[str, Any]], output_dir: Path, palette: list[str]) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    splits = ["train", "val", "test"]
    bal_acc = [float(split_metrics[split_name]["balanced_accuracy"]) for split_name in splits]
    macro_f1 = [float(split_metrics[split_name]["macro_f1"]) for split_name in splits]
    losses = [float(split_metrics[split_name]["loss"]) for split_name in splits]
    ece = [float(split_metrics[split_name]["ece"]) for split_name in splits]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    x = np.arange(len(splits))
    width = 0.34

    axes[0].bar(x - (width / 2), bal_acc, width=width, color=palette[0], label="Balanced accuracy")
    axes[0].bar(x + (width / 2), macro_f1, width=width, color=palette[1], label="Macro F1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Split Classification Metrics")
    axes[0].legend(frameon=False)

    axes[1].bar(x - (width / 2), losses, width=width, color=palette[2], label="Loss")
    axes[1].bar(x + (width / 2), ece, width=width, color=palette[3], label="ECE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].set_title("Split Loss and Calibration")
    axes[1].legend(frameon=False)

    _save_figure(fig, output_dir, FIGURE_FILENAMES["split_overview"])
    plt.close(fig)


def _generate_xai_gallery(
    train_config: dict[str, Any],
    checkpoint_path: Path,
    selected_rows: list[dict[str, Any]],
    class_names: list[str],
    output_dir: Path,
    device: str,
) -> list[str]:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore

    model = create_model(train_config["model"], num_classes=len(class_names)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    disable_inplace_relu(model)
    model.eval()

    camera = GradCAM(model, get_gradcam_target_layer(model, str(train_config["model"]["name"])))
    ncols = 3
    nrows = max(1, int(np.ceil(len(selected_rows) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    summary_lines: list[str] = []

    for axis in axes_array.ravel():
        axis.axis("off")

    for idx, row in enumerate(selected_rows):
        axis = axes_array.ravel()[idx]
        image_tensor = load_image_for_inference(
            image_path=row["raw_image_path"],
            preprocessing_mode=str(train_config.get("preprocessing_mode", "raw_rgb")),
            cornea_mask_path=row.get("cornea_mask_path"),
            image_size=int(train_config.get("image_size", 224)),
        ).to(device)
        predicted_index = class_names.index(str(row["pred_label"]))
        cam_array = camera.generate(image_tensor, predicted_index)
        raw_image = safe_open_image(Path(str(row["raw_image_path"])))
        overlay = overlay_cam_on_image(raw_image, cam_array)
        axis.imshow(overlay)
        axis.set_title(
            f"{row['image_id']} | pred={row['pred_label']}\ntrue={row['true_label']} | conf={float(row['confidence']):.3f}",
            fontsize=8,
        )
        axis.axis("off")
        summary_lines.append(
            f"- image_id={row['image_id']} pred={row['pred_label']} true={row['true_label']} confidence={float(row['confidence']):.4f}"
        )

    camera.close()
    _save_figure(fig, output_dir, FIGURE_FILENAMES["xai"])
    plt.close(fig)
    return summary_lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging()
    train_config = resolve_config(args.train_config)
    task_config = resolve_config(train_config["task_config"])
    task_definition = get_task_definition(str(task_config["task_name"]))
    experiment_name = build_experiment_name({**train_config, "task_name": task_definition.task_name})
    output_dir = Path(args.output_root) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = Path(
        args.predictions_csv
        or Path(train_config.get("output_root", "outputs")) / "predictions" / experiment_name / "test_predictions.csv"
    )
    metrics_json = Path(
        args.metrics_json
        or Path(train_config.get("output_root", "outputs")) / "metrics" / experiment_name / "test_metrics.json"
    )
    split_metrics = {
        split_name: json.loads(
            (
                Path(train_config.get("output_root", "outputs")) / "metrics" / experiment_name / f"{split_name}_metrics.json"
            ).read_text(encoding="utf-8")
        )
        for split_name in ("train", "val", "test")
    }

    prediction_rows = _load_predictions(predictions_csv)
    test_metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    class_names = list(task_definition.class_names)
    probability_columns = [f"prob_{class_name}" for class_name in class_names]
    y_true = [class_names.index(str(row["true_label"])) for row in prediction_rows]
    y_pred = [class_names.index(str(row["pred_label"])) for row in prediction_rows]
    probabilities = [[float(row[column]) for column in probability_columns] for row in prediction_rows]
    recomputed = compute_classification_metrics(y_true, y_pred, probabilities, class_names)

    palette = _configure_publication_style()
    confusion_matrix = [
        [sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == row_idx and pred == col_idx) for col_idx in range(len(class_names))]
        for row_idx in range(len(class_names))
    ]
    _plot_confusion_matrix(confusion_matrix, class_names, output_dir, palette)
    _plot_roc_pr_curves(y_true, probabilities, class_names, output_dir, palette)
    reliability_rows = build_reliability_bins(probabilities, y_true)
    _plot_reliability(reliability_rows, float(test_metrics["ece"]), output_dir)
    _plot_confidence_histogram(
        [float(row["confidence"]) for row in prediction_rows],
        [_to_bool(row["correct"]) for row in prediction_rows],
        output_dir,
        palette,
    )
    _plot_per_class_metrics(test_metrics["classification_report"], class_names, output_dir, palette)
    _plot_split_overview(split_metrics, output_dir, palette)
    xai_rows = select_xai_rows(prediction_rows, class_names, max(1, int(args.xai_count)))
    device = resolve_device(args.device)
    if device == "cuda":
        logger.warning("CUDA requested for paper figures, but this environment is currently unstable; using CPU instead.")
        device = "cpu"
    xai_summary_lines = _generate_xai_gallery(train_config, Path(args.checkpoint), xai_rows, class_names, output_dir, device)

    summary_lines = [
        f"# Paper Figure Bundle: {experiment_name}",
        "",
        "## Test Metrics",
        "",
        f"- Accuracy: {float(test_metrics['accuracy']):.4f}",
        f"- Balanced accuracy: {float(test_metrics['balanced_accuracy']):.4f}",
        f"- Macro F1: {float(test_metrics['macro_f1']):.4f}",
        f"- Weighted F1: {float(test_metrics['weighted_f1']):.4f}",
        f"- ROC-AUC macro OVR: {float(test_metrics['roc_auc_macro_ovr']):.4f}",
        f"- PR-AUC macro OVR: {float(test_metrics['pr_auc_macro_ovr']):.4f}",
        f"- ECE: {float(test_metrics['ece']):.4f}",
        "",
        "## Generated Figures",
        "",
    ]
    for key in ("confusion", "roc", "pr", "reliability", "confidence", "per_class", "split_overview", "xai"):
        stem = FIGURE_FILENAMES[key]
        summary_lines.append(f"- `{stem}.png`")
        summary_lines.append(f"- `{stem}.pdf`")
    summary_lines.extend(
        [
            "",
            "## Grad-CAM Examples",
            "",
            *xai_summary_lines,
            "",
            "## Recomputed Metric Cross-Check",
            "",
            f"- Balanced accuracy: {float(recomputed['metrics']['balanced_accuracy']):.4f}",
            f"- Macro F1: {float(recomputed['metrics']['macro_f1']):.4f}",
        ]
    )
    write_text(output_dir / "figure_manifest.md", "\n".join(summary_lines))
    logger.info("Saved paper figure bundle for %s to %s", experiment_name, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
