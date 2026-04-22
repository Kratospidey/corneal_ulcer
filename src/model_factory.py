from __future__ import annotations

from typing import Any


def create_model(model_config: dict[str, Any], num_classes: int):
    import torch  # type: ignore
    from torchvision.models import AlexNet_Weights, ResNet18_Weights, VGG16_Weights, alexnet, resnet18, vgg16  # type: ignore

    model_name = str(model_config["name"]).lower()
    pretrained = bool(model_config.get("pretrained", True))
    freeze_backbone = bool(model_config.get("freeze_backbone", False))
    drop_path_rate = float(model_config.get("drop_path_rate", 0.0))
    drop_rate = float(model_config.get("drop_rate", 0.0))

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name == "vgg16":
        weights = VGG16_Weights.DEFAULT if pretrained else None
        model = vgg16(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    elif model_name == "alexnet":
        weights = AlexNet_Weights.DEFAULT if pretrained else None
        model = alexnet(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    elif model_name.startswith("convnextv2"):
        import timm  # type: ignore

        timm_kwargs = {
            "pretrained": pretrained,
            "num_classes": num_classes,
        }
        if drop_path_rate > 0.0:
            timm_kwargs["drop_path_rate"] = drop_path_rate
        if drop_rate > 0.0:
            timm_kwargs["drop_rate"] = drop_rate
        model = timm.create_model(model_name, **timm_kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        freeze_feature_extractor(model, model_name)
    return model


def load_backbone_checkpoint(model, checkpoint_path: str | None) -> dict[str, Any]:
    if not checkpoint_path:
        return {"loaded": False, "missing_keys": [], "unexpected_keys": []}

    import torch  # type: ignore

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_state = checkpoint.get("backbone_state_dict") or checkpoint.get("model_state_dict") or checkpoint
    target = model.backbone if hasattr(model, "backbone") else model
    prefix = "backbone."
    target_state = target.state_dict()
    filtered_state = {}
    skipped_incompatible_keys: list[str] = []
    for key, value in raw_state.items():
        normalized_key = key[len(prefix) :] if key.startswith(prefix) else key
        if normalized_key.startswith("head.fc.") or normalized_key.startswith("head.weight") or normalized_key.startswith("head.bias"):
            skipped_incompatible_keys.append(normalized_key)
            continue
        if normalized_key not in target_state:
            continue
        if hasattr(value, "shape") and tuple(target_state[normalized_key].shape) != tuple(value.shape):
            skipped_incompatible_keys.append(normalized_key)
            continue
        filtered_state[normalized_key] = value
    if not filtered_state:
        return {
            "loaded": False,
            "missing_keys": [],
            "unexpected_keys": [],
            "skipped_incompatible_keys": skipped_incompatible_keys,
            "reason": "no_compatible_backbone_keys",
        }
    missing_keys, unexpected_keys = target.load_state_dict(filtered_state, strict=False)
    return {
        "loaded": True,
        "loaded_key_count": len(filtered_state),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "skipped_incompatible_keys": skipped_incompatible_keys,
    }


def freeze_feature_extractor(model, model_name: str) -> None:
    if model_name == "resnet18":
        for name, parameter in model.named_parameters():
            if not name.startswith("fc."):
                parameter.requires_grad = False
    elif model_name in {"vgg16", "alexnet"}:
        for name, parameter in model.named_parameters():
            if not name.startswith("classifier.6"):
                parameter.requires_grad = False
    elif model_name.startswith("convnextv2"):
        for name, parameter in model.named_parameters():
            if not name.startswith("head."):
                parameter.requires_grad = False
    else:
        raise ValueError(f"Unsupported model for freezing: {model_name}")


def get_gradcam_target_layer(model, model_name: str):
    model_name = model_name.lower()
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "vgg16":
        return model.features[28]
    if model_name == "alexnet":
        return model.features[10]
    if model_name.startswith("convnextv2"):
        return model.stages[-1].blocks[-1].conv_dw
    raise ValueError(f"No Grad-CAM target layer for {model_name}")
