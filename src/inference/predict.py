from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_utils import resolve_config
from data.label_utils import get_task_definition, index_to_class
from experiment_utils import resolve_device, setup_logging
from inference.inference_utils import load_image_for_inference
from model_factory import create_model


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Single-image inference for Stage 3 baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--cornea-mask-path")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging()
    config = resolve_config(args.config)
    task_config = resolve_config(config["task_config"])
    task_definition = get_task_definition(str(task_config["task_name"]))
    label_map = index_to_class(task_definition.class_names)
    device = resolve_device(args.device)

    model = create_model(config["model"], num_classes=len(task_definition.class_names)).to(device)
    import torch  # type: ignore

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_tensor = load_image_for_inference(
        image_path=args.image_path,
        preprocessing_mode=str(config.get("preprocessing_mode", "raw_rgb")),
        cornea_mask_path=args.cornea_mask_path,
        image_size=int(config.get("image_size", 224)),
    ).to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(image_tensor), dim=1)[0].cpu().tolist()
    pred_index = max(range(len(probs)), key=lambda index: probs[index])
    logger.info("Prediction: %s (confidence=%.4f)", label_map[pred_index], probs[pred_index])
    for index, prob in enumerate(probs):
        logger.info("%s: %.4f", label_map[index], prob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
