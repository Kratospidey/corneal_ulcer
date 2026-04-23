from __future__ import annotations

from argparse import ArgumentParser

import torch  # type: ignore

from config_utils import resolve_config, write_json
from experiment_utils import prepare_output_dirs, resolve_device, set_seed, setup_logging
from external_data.slid import (
    build_slid_cornea_dataloaders,
    build_slid_cornea_datasets,
    ensure_slid_images_extracted,
    load_slid_manifest,
    load_slid_split,
    resolve_slid_manifest_paths,
    slid_split_summary,
)
from segmentation.training import run_segmentation_training
from segmentation.transforms import build_segmentation_transforms


class ConvNeXtV2CorneaPretrainModel(torch.nn.Module):
    def __init__(self, backbone_name: str = "convnextv2_tiny", pretrained: bool = True, out_channels: int = 2) -> None:
        super().__init__()
        import timm  # type: ignore

        self.backbone_name = str(backbone_name)
        self.backbone = timm.create_model(self.backbone_name, pretrained=pretrained, num_classes=3)
        self.segmentation_head = torch.nn.Sequential(
            torch.nn.Conv2d(768, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
            torch.nn.Conv2d(256, out_channels, kernel_size=1),
        )

    def forward(self, inputs):
        features = self.backbone.forward_features(inputs)
        logits = self.segmentation_head(features)
        return torch.nn.functional.interpolate(
            logits,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.backbone.state_dict().items()
            if not key.startswith("head.")
        }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train the SLID cornea-mask external pretraining stage.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging()
    config = resolve_config(args.config)

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(args.device)
    experiment_name = str(config["experiment_name"])
    output_dirs = prepare_output_dirs(experiment_name, output_root=config.get("output_root", "outputs"))

    extracted_dir = ensure_slid_images_extracted(
        config["image_zip_path"],
        config["extracted_image_dir"],
        logger=logger,
    )
    manifest_df = resolve_slid_manifest_paths(
        load_slid_manifest(config["manifest_path"]),
        extracted_dir,
    )
    split_df = load_slid_split(config["split_file"])
    transforms_by_split = build_segmentation_transforms(int(config.get("image_size", 224)))
    datasets = build_slid_cornea_datasets(manifest_df, split_df, transforms_by_split)
    loaders = build_slid_cornea_dataloaders(
        datasets=datasets,
        batch_size=int(config.get("batch_size", 8)),
        num_workers=int(config.get("num_workers", 4)),
    )
    split_summary = slid_split_summary(datasets)

    model = ConvNeXtV2CorneaPretrainModel(
        backbone_name=str(config.get("model", {}).get("name", "convnextv2_tiny")),
        pretrained=bool(config.get("model", {}).get("pretrained", True)),
        out_channels=2,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config.get("epochs", 12))),
    )

    summary = run_segmentation_training(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dirs=output_dirs,
        training_config=config,
        experiment_name=experiment_name,
        split_summary=split_summary,
    )
    checkpoint_path = output_dirs["models"] / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["backbone_state_dict"] = model.backbone_state_dict()
    checkpoint["external_pretrain"] = {
        "dataset": "SLID",
        "task": "cornea_mask",
        "backbone_name": model.backbone_name,
    }
    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, output_dirs["exported"] / "best.pt")
    write_json(
        output_dirs["reports"] / "run_context.json",
        {
            "config_path": args.config,
            "device": device,
            "split_file": str(config["split_file"]),
            "split_summary": split_summary,
            "best_checkpoint": str(checkpoint_path),
            "exported_checkpoint": str(output_dirs["exported"] / "best.pt"),
            "dataset": "SLID",
            "task": "cornea_mask",
            "extracted_image_dir": str(extracted_dir),
        },
    )
    logger.info(
        "Saved external pretraining checkpoint %s and exported backbone checkpoint %s",
        checkpoint_path,
        output_dirs["exported"] / "best.pt",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
