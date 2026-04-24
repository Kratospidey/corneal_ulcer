from __future__ import annotations

from typing import Any


class FocalLoss:
    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = float(label_smoothing)

    def __call__(self, logits, targets):
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        weight = self.weight.to(logits.device) if self.weight is not None else None
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class LogitAdjustedCrossEntropyLoss:
    def __init__(self, logit_adjustment, weight=None, label_smoothing: float = 0.0) -> None:
        self.logit_adjustment = logit_adjustment
        self.weight = weight
        self.label_smoothing = float(label_smoothing)

    def __call__(self, logits, targets):
        import torch.nn.functional as F  # type: ignore

        weight = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(
            logits + self.logit_adjustment.to(logits.device),
            targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )


class ClassBalancedFocalLoss(FocalLoss):
    pass


def compute_class_priors(class_names: list[str] | tuple[str, ...], label_counts: dict[str, int]):
    import torch  # type: ignore

    total = float(sum(label_counts.values()) or 1)
    priors = [max(1.0 / total, float(label_counts.get(class_name, 0)) / total) for class_name in class_names]
    return torch.tensor(priors, dtype=torch.float32)


def compute_logit_adjustment(
    class_names: list[str] | tuple[str, ...],
    label_counts: dict[str, int],
    tau: float = 1.0,
):
    import torch  # type: ignore

    priors = compute_class_priors(class_names, label_counts)
    return torch.log(priors).to(dtype=torch.float32) * float(tau)


def compute_effective_number_weights(
    class_names: list[str] | tuple[str, ...],
    label_counts: dict[str, int],
    beta: float = 0.999,
    normalize: bool = True,
):
    import torch  # type: ignore

    weights = []
    beta = float(beta)
    for class_name in class_names:
        count = max(1, int(label_counts.get(class_name, 0)))
        effective_num = 1.0 - (beta ** count)
        weight = (1.0 - beta) / max(effective_num, 1e-12)
        weights.append(weight)
    tensor = torch.tensor(weights, dtype=torch.float32)
    if normalize and tensor.mean().item() > 0:
        tensor = tensor / tensor.mean()
    return tensor


def ordinal_targets(targets, num_classes: int):
    import torch  # type: ignore

    thresholds = torch.arange(num_classes - 1, device=targets.device, dtype=targets.dtype)
    return (targets.unsqueeze(1) > thresholds.unsqueeze(0)).to(dtype=torch.float32)


def compute_ordinal_aux_loss(ordinal_logits, targets, num_classes: int):
    import torch  # type: ignore

    loss_fn = torch.nn.BCEWithLogitsLoss()
    return loss_fn(ordinal_logits, ordinal_targets(targets, num_classes=num_classes))


def build_loss(
    loss_name: str,
    class_weights=None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    class_names: list[str] | tuple[str, ...] | None = None,
    label_counts: dict[str, int] | None = None,
    logit_adjustment_tau: float = 1.0,
    class_balanced_beta: float = 0.999,
):
    import torch  # type: ignore

    name = loss_name.lower()
    if name in {"ce", "cross_entropy"}:
        return torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    if name in {"weighted_ce", "weighted_cross_entropy"}:
        return torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(label_smoothing))
    if name == "focal":
        return FocalLoss(weight=class_weights, gamma=focal_gamma, label_smoothing=label_smoothing)
    if name == "logit_adjusted_ce":
        if class_names is None or label_counts is None:
            raise ValueError("logit_adjusted_ce requires class_names and label_counts.")
        return LogitAdjustedCrossEntropyLoss(
            logit_adjustment=compute_logit_adjustment(class_names, label_counts, tau=logit_adjustment_tau),
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    if name == "class_balanced_focal":
        if class_names is None or label_counts is None:
            raise ValueError("class_balanced_focal requires class_names and label_counts.")
        weights = compute_effective_number_weights(class_names, label_counts, beta=class_balanced_beta)
        return ClassBalancedFocalLoss(weight=weights, gamma=focal_gamma, label_smoothing=label_smoothing)
    raise ValueError(f"Unsupported loss: {loss_name}")


def compute_class_weights(class_names: list[str] | tuple[str, ...], label_counts: dict[str, int], normalize: bool = True):
    import torch  # type: ignore

    weights = []
    total = sum(label_counts.values()) or 1
    for class_name in class_names:
        count = max(1, label_counts.get(class_name, 0))
        weights.append(total / (len(class_names) * count))
    tensor = torch.tensor(weights, dtype=torch.float32)
    if normalize and tensor.sum().item() > 0:
        tensor = tensor / tensor.mean()
    return tensor
