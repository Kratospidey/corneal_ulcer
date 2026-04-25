from __future__ import annotations

from typing import Any


def _split_parameter_groups(model):
    head_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(
            (
                "fc.",
                "classifier.",
                "head.",
                "feature_norm.",
                "ordinal_head.",
                "early_norm.",
                "early_proj.",
                "final_proj.",
            )
        ):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    return backbone_parameters, head_parameters


def _build_prefix_parameter_groups(model, group_specs: list[dict[str, Any]], default_weight_decay: float):
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assigned_ids: set[int] = set()
    parameter_groups = []

    for group_spec in group_specs:
        prefixes = tuple(str(prefix) for prefix in group_spec.get("prefixes", []) if str(prefix).strip())
        if not prefixes:
            continue
        params = []
        for name, parameter in named_parameters:
            if id(parameter) in assigned_ids:
                continue
            if any(name.startswith(prefix) for prefix in prefixes):
                params.append(parameter)
                assigned_ids.add(id(parameter))
        if not params:
            continue
        parameter_groups.append(
            {
                "params": params,
                "lr": float(group_spec["lr"]),
                "weight_decay": float(group_spec.get("weight_decay", default_weight_decay)),
            }
        )

    remaining = [parameter for _, parameter in named_parameters if id(parameter) not in assigned_ids]
    if remaining:
        fallback_lr = float(group_specs[-1]["lr"]) if group_specs else 0.0
        parameter_groups.append(
            {
                "params": remaining,
                "lr": fallback_lr,
                "weight_decay": default_weight_decay,
            }
        )
    return parameter_groups


def build_optimizer(model, training_config: dict[str, Any]):
    import torch  # type: ignore

    optimizer_name = str(training_config.get("optimizer", "adamw")).lower()
    lr = float(training_config.get("lr", 3e-4))
    weight_decay = float(training_config.get("weight_decay", 1e-4))
    optimizer_groups = list(training_config.get("optimizer_param_groups", []) or [])
    if optimizer_groups:
        parameters = _build_prefix_parameter_groups(model, optimizer_groups, default_weight_decay=weight_decay)
    else:
        backbone_parameters, head_parameters = _split_parameter_groups(model)
        head_lr_mult = float(training_config.get("head_lr_mult", 1.0))
        backbone_lr = float(training_config.get("backbone_lr", lr))
        head_lr = float(training_config.get("head_lr", lr * head_lr_mult))

        if head_parameters and (
            head_lr != backbone_lr or head_lr_mult != 1.0 or "head_lr" in training_config or "backbone_lr" in training_config
        ):
            parameters = [
                {"params": backbone_parameters, "lr": backbone_lr, "weight_decay": weight_decay},
                {"params": head_parameters, "lr": head_lr, "weight_decay": weight_decay},
            ]
        else:
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = float(training_config.get("momentum", 0.9))
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer, training_config: dict[str, Any]):
    import torch  # type: ignore

    scheduler_name = str(training_config.get("scheduler", "cosine")).lower()
    epochs = int(training_config.get("epochs", 10))

    if scheduler_name in {"none", "off"}:
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name == "step":
        step_size = int(training_config.get("step_size", max(1, epochs // 3)))
        gamma = float(training_config.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if scheduler_name == "plateau":
        factor = float(training_config.get("factor", 0.5))
        patience = int(training_config.get("scheduler_patience", 2))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=patience)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def step_scheduler(scheduler, monitor_value: float | None = None) -> None:
    if scheduler is None:
        return
    class_name = scheduler.__class__.__name__.lower()
    if "reducelronplateau" in class_name:
        scheduler.step(monitor_value)
    else:
        scheduler.step()
