from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_utils import resolve_config, write_text
from data.label_utils import get_task_definition
from experiment_utils import build_experiment_name, prepare_output_dirs, resolve_device, setup_logging
from explainability.gradcam_utils import GradCAM, disable_inplace_relu, save_overlay
from inference.inference_utils import load_image_for_inference
from model_factory import create_model, get_gradcam_target_layer
from utils_io import safe_open_image


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Grad-CAM explainability for Stage 3 CNN baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-config")
    parser.add_argument("--predictions-csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging()
    explain_config = resolve_config(args.config)
    train_config_path = args.train_config or explain_config["train_config"]
    train_config = resolve_config(train_config_path)
    task_config = resolve_config(train_config["task_config"])
    task_definition = get_task_definition(str(task_config["task_name"]))
    experiment_name = build_experiment_name({**train_config, "task_name": task_definition.task_name})
    output_dirs = prepare_output_dirs(experiment_name, output_root=train_config.get("output_root", "outputs"))
    device = resolve_device(args.device)

    model = create_model(train_config["model"], num_classes=len(task_definition.class_names)).to(device)
    import pandas as pd  # type: ignore
    import torch  # type: ignore

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    disable_inplace_relu(model)
    model.eval()

    predictions_csv = Path(args.predictions_csv or explain_config.get("predictions_csv", output_dirs["predictions"] / "test_predictions.csv"))
    predictions_df = pd.read_csv(predictions_csv)
    selected_rows = select_examples(predictions_df, explain_config)
    cam = GradCAM(model, get_gradcam_target_layer(model, str(train_config["model"]["name"])))
    summary_lines = ["# Baseline Explainability Summary", ""]

    for category, rows in selected_rows.items():
        category_dir = output_dirs["explainability"] / category
        category_dir.mkdir(parents=True, exist_ok=True)
        summary_lines.append(f"## {category.title()}")
        summary_lines.append("")
        for _, row in rows.iterrows():
            image_tensor = load_image_for_inference(
                image_path=row["raw_image_path"],
                preprocessing_mode=str(train_config.get("preprocessing_mode", "raw_rgb")),
                cornea_mask_path=row.get("cornea_mask_path"),
                image_size=int(train_config.get("image_size", 224)),
            ).to(device)
            predicted_index = task_definition.class_names.index(str(row["pred_label"]))
            cam_array = cam.generate(image_tensor, predicted_index)
            image = safe_open_image(Path(row["raw_image_path"]))
            output_path = category_dir / f"{row['image_id']}.png"
            save_overlay(
                image=image,
                cam_array=cam_array,
                output_path=output_path,
                title=f"{row['image_id']} pred={row['pred_label']} true={row['true_label']}",
            )
            summary_lines.append(f"- {row['image_id']}: pred={row['pred_label']} true={row['true_label']} confidence={row['confidence']}")
        summary_lines.append("")
    cam.close()
    write_text(output_dirs["reports"] / "explainability_summary.md", "\n".join(summary_lines))
    logger.info("Saved Grad-CAM outputs for %s", experiment_name)
    return 0


def select_examples(predictions_df, explain_config):
    per_bucket = int(explain_config.get("examples_per_bucket", 6))
    correct_mask = predictions_df["correct"].astype(str).str.lower().isin(["true", "1"])
    correct_df = predictions_df[correct_mask].sort_values("confidence", ascending=False).head(per_bucket)
    incorrect_df = predictions_df[~correct_mask].sort_values("confidence", ascending=False).head(per_bucket)
    borderline_correct = predictions_df[correct_mask].sort_values("confidence", ascending=True).head(max(1, per_bucket // 2))
    borderline_incorrect = predictions_df[~correct_mask].sort_values("confidence", ascending=False).head(
        max(1, per_bucket - len(borderline_correct))
    )
    borderline_df = (
        predictions_df.iloc[0:0]
        if borderline_correct.empty and borderline_incorrect.empty
        else __import__("pandas").concat([borderline_correct, borderline_incorrect], ignore_index=True)
    )
    return {"correct": correct_df, "incorrect": incorrect_df, "borderline": borderline_df}


if __name__ == "__main__":
    raise SystemExit(main())
