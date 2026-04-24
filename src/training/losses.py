from __future__ import annotations

from typing import Any


class FocalLoss:
    def __init__(self, weight=None, gamma: float = 2.0) -> None:
        import torch  # type: ignore

        self.weight = weight
        self.gamma = gamma
        self._ce = torch.nn.CrossEntropyLoss(weight=weight, reduction="none")

    def __call__(self, logits, targets):
        import torch  # type: ignore

        ce_loss = self._ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


def build_loss(loss_name: str, class_weights=None, focal_gamma: float = 2.0, label_smoothing: float = 0.0):
    import torch  # type: ignore

    name = loss_name.lower()
    if name in {"ce", "cross_entropy"}:
        return torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    if name in {"weighted_ce", "weighted_cross_entropy"}:
        return torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(label_smoothing))
    if name == "focal":
        return FocalLoss(weight=class_weights, gamma=focal_gamma)
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
