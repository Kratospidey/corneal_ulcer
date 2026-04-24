from __future__ import annotations

from pathlib import Path
from typing import Any

import torch  # type: ignore

from proxy_signal.geometry import DEFAULT_TARGET_NAMES, resolve_proxy_geometry_config


def create_model(model_config: dict[str, Any], num_classes: int):
    from torchvision.models import AlexNet_Weights, ResNet18_Weights, VGG16_Weights, alexnet, resnet18, vgg16  # type: ignore

    model_name = str(model_config["name"]).lower()
    pretrained = bool(model_config.get("pretrained", True))
    freeze_backbone = bool(model_config.get("freeze_backbone", False))
    drop_path_rate = float(model_config.get("drop_path_rate", 0.0))
    drop_rate = float(model_config.get("drop_rate", 0.0))
    proxy_geometry_aux_config = resolve_proxy_geometry_config(dict(model_config.get("proxy_geometry_aux", {})))
    proxy_geometry_aux_enabled = bool(proxy_geometry_aux_config.get("enabled", False))

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
    elif model_name.startswith(("convnextv2", "swin", "maxvit")):
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
        if freeze_backbone:
            freeze_feature_extractor(model, model_name)
        if proxy_geometry_aux_enabled:
            if not model_name.startswith("convnextv2"):
                raise ValueError("proxy_geometry_aux is only supported for convnextv2 backbones.")
            model = ConvNeXtProxyGeometryAuxModel(
                backbone=model,
                target_dim=len(proxy_geometry_aux_config.get("target_names") or DEFAULT_TARGET_NAMES),
                hidden_dim=int(proxy_geometry_aux_config.get("hidden_dim", 192)),
            )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone and not model_name.startswith(("convnextv2", "swin", "maxvit")):
        freeze_feature_extractor(model, model_name)
    return model


class ConvNeXtProxyGeometryAuxModel(torch.nn.Module):
    def __init__(self, backbone, target_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.backbone = backbone
        self.target_dim = int(target_dim)
        self.proxy_geometry_head = torch.nn.Sequential(
            torch.nn.Linear(int(self.backbone.num_features), int(hidden_dim)),
            torch.nn.LayerNorm(int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(hidden_dim), self.target_dim),
        )

    def _forward_logits_and_embedding(self, inputs):
        features = self.backbone.forward_features(inputs)
        logits = self.backbone.forward_head(features, pre_logits=False)
        embedding = self.backbone.forward_head(features, pre_logits=True)
        return logits, embedding

    def forward(self, inputs):
        logits, _ = self._forward_logits_and_embedding(inputs)
        return logits

    def forward_with_proxy(self, inputs):
        logits, embedding = self._forward_logits_and_embedding(inputs)
        proxy_geometry = torch.sigmoid(self.proxy_geometry_head(embedding))
        return logits, proxy_geometry


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
    elif model_name.startswith(("swin", "maxvit")):
        for name, parameter in model.named_parameters():
            if not name.startswith("head."):
                parameter.requires_grad = False
    else:
        raise ValueError(f"Unsupported model for freezing: {model_name}")


def _extract_backbone_state_dict(checkpoint_payload: dict[str, Any]) -> dict[str, Any]:
    if "backbone_state_dict" in checkpoint_payload:
        return dict(checkpoint_payload["backbone_state_dict"])
    state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
    return {key: value for key, value in state_dict.items() if not key.startswith("head.")}


def load_backbone_warmstart(model, checkpoint_path: str | Path) -> dict[str, Any]:
    import torch  # type: ignore

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    backbone_state = _extract_backbone_state_dict(checkpoint_payload)
    load_target = model.backbone if hasattr(model, "forward_with_proxy") and hasattr(model, "backbone") else model
    target_state = load_target.state_dict()

    if not backbone_state:
        raise ValueError(f"No backbone weights found in warm-start checkpoint: {checkpoint_path}")

    unexpected = [key for key in backbone_state if key not in target_state]
    if unexpected:
        raise ValueError(
            f"Warm-start checkpoint contains unexpected backbone keys for this model: {unexpected[:10]}"
        )

    mismatched = []
    for key, value in backbone_state.items():
        if tuple(target_state[key].shape) != tuple(value.shape):
            mismatched.append((key, tuple(value.shape), tuple(target_state[key].shape)))
    if mismatched:
        preview = ", ".join(f"{key}: {src} -> {dst}" for key, src, dst in mismatched[:10])
        raise ValueError(f"Warm-start backbone shape mismatch: {preview}")

    missing_backbone = [
        key
        for key in target_state
        if not key.startswith("head.") and key not in backbone_state
    ]
    if missing_backbone:
        raise ValueError(
            f"Warm-start checkpoint is missing required backbone keys: {missing_backbone[:10]}"
        )

    missing_keys, unexpected_keys = load_target.load_state_dict(backbone_state, strict=False)
    bad_missing = [key for key in missing_keys if not key.startswith("head.")]
    if bad_missing or unexpected_keys:
        raise ValueError(
            f"Warm-start load produced invalid missing/unexpected keys: missing={bad_missing[:10]}, "
            f"unexpected={unexpected_keys[:10]}"
        )

    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded_backbone_keys": len(backbone_state),
        "missing_head_keys": list(missing_keys),
        "external_pretrain": checkpoint_payload.get("external_pretrain", {}),
    }


def get_gradcam_target_layer(model, model_name: str):
    model_name = model_name.lower()
    if hasattr(model, "forward_with_proxy") and hasattr(model, "backbone"):
        model = model.backbone
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "vgg16":
        return model.features[28]
    if model_name == "alexnet":
        return model.features[10]
    if model_name.startswith("convnextv2"):
        return model.stages[-1].blocks[-1].conv_dw
    if model_name.startswith("swin"):
        return model.layers[-1].blocks[-1].norm1
    raise ValueError(f"No Grad-CAM target layer for {model_name}")
