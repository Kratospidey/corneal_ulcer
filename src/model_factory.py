from __future__ import annotations

from typing import Any

def create_model(model_config: dict[str, Any], num_classes: int):
    import timm  # type: ignore

    model_name = str(model_config["name"]).lower()
    pretrained = bool(model_config.get("pretrained", True))
    freeze_backbone = bool(model_config.get("freeze_backbone", False))
    drop_path_rate = float(model_config.get("drop_path_rate", 0.0))
    drop_rate = float(model_config.get("drop_rate", 0.0))
    if not model_name.startswith("convnextv2"):
        raise ValueError(f"Unsupported model: {model_name}")

    timm_kwargs: dict[str, Any] = {
        "pretrained": pretrained,
        "num_classes": num_classes,
    }
    if drop_path_rate > 0.0:
        timm_kwargs["drop_path_rate"] = drop_path_rate
    if drop_rate > 0.0:
        timm_kwargs["drop_rate"] = drop_rate
    model = timm.create_model(model_name, **timm_kwargs)
    if freeze_backbone:
        freeze_feature_extractor(model, model_name)
    return model


def freeze_feature_extractor(model, model_name: str) -> None:
    if model_name.startswith("convnextv2"):
        for name, parameter in model.named_parameters():
            if not name.startswith("head."):
                parameter.requires_grad = False
    else:
        raise ValueError(f"Unsupported model for freezing: {model_name}")


def get_gradcam_target_layer(model, model_name: str):
    model_name = model_name.lower()
    if model_name.startswith("convnextv2"):
        return model.stages[-1].blocks[-1].conv_dw
    raise ValueError(f"No Grad-CAM target layer for {model_name}")
