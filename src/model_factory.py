from __future__ import annotations

from typing import Any
import torch


def _stage_feature_channels(model) -> list[int]:
    feature_info = getattr(model, "feature_info", None)
    if feature_info is not None and hasattr(feature_info, "channels"):
        return [int(channel) for channel in feature_info.channels()]
    return [96, 192, 384, 768]


class ConvNeXtV2Phase1Model(torch.nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        pretrained: bool,
        num_classes: int,
        drop_path_rate: float,
        drop_rate: float,
        freeze_backbone: bool,
        multiscale_config: dict[str, Any] | None = None,
        ordinal_config: dict[str, Any] | None = None,
    ) -> None:
        import timm  # type: ignore

        super().__init__()
        backbone_kwargs: dict[str, Any] = {
            "pretrained": pretrained,
            "num_classes": 0,
            "global_pool": "",
        }
        if drop_path_rate > 0.0:
            backbone_kwargs["drop_path_rate"] = drop_path_rate
        if drop_rate > 0.0:
            backbone_kwargs["drop_rate"] = drop_rate
        self.backbone = timm.create_model(model_name, **backbone_kwargs)
        self.num_classes = int(num_classes)
        self.feature_dim = int(getattr(self.backbone, "num_features", 768))
        self.multiscale_config = dict(multiscale_config or {})
        self.ordinal_config = dict(ordinal_config or {})
        self.enable_multiscale = bool(self.multiscale_config.get("enabled", False))
        self.enable_ordinal = bool(self.ordinal_config.get("enabled", False))

        channels = _stage_feature_channels(self.backbone)
        head_dropout = float(self.multiscale_config.get("dropout", 0.0))
        fusion_dim = int(self.multiscale_config.get("fusion_dim", 256))
        stage_index = int(self.multiscale_config.get("stage_index", 2))
        self.multiscale_stage_index = max(0, min(stage_index, len(channels) - 2))
        self.feature_norm = torch.nn.LayerNorm(self.feature_dim)
        if self.enable_multiscale:
            early_dim = int(channels[self.multiscale_stage_index])
            self.early_norm = torch.nn.LayerNorm(early_dim)
            self.early_proj = torch.nn.Linear(early_dim, fusion_dim)
            self.final_proj = torch.nn.Linear(self.feature_dim, fusion_dim)
            self.fusion_act = torch.nn.GELU()
            self.head_dropout = torch.nn.Dropout(head_dropout)
            classifier_in_dim = fusion_dim * 2
        else:
            self.early_norm = None
            self.early_proj = None
            self.final_proj = None
            self.fusion_act = None
            self.head_dropout = torch.nn.Dropout(head_dropout) if head_dropout > 0.0 else torch.nn.Identity()
            classifier_in_dim = self.feature_dim
        self.classifier = torch.nn.Linear(classifier_in_dim, num_classes)
        self.ordinal_head = torch.nn.Linear(classifier_in_dim, num_classes - 1) if self.enable_ordinal else None
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, inputs):
        x = self.backbone.stem(inputs)
        stage_outputs = []
        for stage in self.backbone.stages:
            x = stage(x)
            stage_outputs.append(x)
        x = self.backbone.norm_pre(x)
        final_vector = x.mean(dim=(-2, -1))
        final_vector = self.feature_norm(final_vector)
        if self.enable_multiscale:
            early_vector = stage_outputs[self.multiscale_stage_index].mean(dim=(-2, -1))
            early_vector = self.early_norm(early_vector)
            fused = torch.cat(
                (
                    self.fusion_act(self.early_proj(early_vector)),
                    self.fusion_act(self.final_proj(final_vector)),
                ),
                dim=1,
            )
            fused = self.head_dropout(fused)
        else:
            fused = self.head_dropout(final_vector)
        logits = self.classifier(fused)
        outputs = {
            "logits": logits,
            "features": fused,
        }
        if self.ordinal_head is not None:
            outputs["ordinal_logits"] = self.ordinal_head(fused)
        return outputs


def model_outputs_to_dict(outputs) -> dict[str, Any]:
    if isinstance(outputs, dict):
        return outputs
    return {"logits": outputs}


def primary_logits(outputs):
    payload = model_outputs_to_dict(outputs)
    return payload["logits"]


def freeze_parameter_prefixes(model, prefixes: list[str] | tuple[str, ...]) -> list[str]:
    normalized = [str(prefix) for prefix in prefixes if str(prefix).strip()]
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in normalized):
            parameter.requires_grad = False
            frozen.append(name)
    return frozen


def create_model(model_config: dict[str, Any], num_classes: int):
    import timm  # type: ignore

    model_name = str(model_config["name"]).lower()
    pretrained = bool(model_config.get("pretrained", True))
    freeze_backbone = bool(model_config.get("freeze_backbone", False))
    drop_path_rate = float(model_config.get("drop_path_rate", 0.0))
    drop_rate = float(model_config.get("drop_rate", 0.0))
    multiscale_config = dict(model_config.get("multiscale_head", {}))
    ordinal_config = dict(model_config.get("ordinal_aux", {}))
    if not model_name.startswith("convnextv2"):
        raise ValueError(f"Unsupported model: {model_name}")

    if bool(multiscale_config.get("enabled", False)) or bool(ordinal_config.get("enabled", False)):
        return ConvNeXtV2Phase1Model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            freeze_backbone=freeze_backbone,
            multiscale_config=multiscale_config,
            ordinal_config=ordinal_config,
        )

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
    if hasattr(model, "backbone") and hasattr(model.backbone, "stages"):
        return model.backbone.stages[-1].blocks[-1].conv_dw
    if model_name.startswith("convnextv2"):
        return model.stages[-1].blocks[-1].conv_dw
    raise ValueError(f"No Grad-CAM target layer for {model_name}")
