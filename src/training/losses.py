from __future__ import annotations

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


class BalancedSoftmaxLoss:
    def __init__(self, class_counts: dict[str, int], class_names: list[str] | tuple[str, ...]) -> None:
        import torch  # type: ignore

        counts = [max(1, int(class_counts.get(class_name, 0))) for class_name in class_names]
        self.class_count_tensor = torch.tensor(counts, dtype=torch.float32)

    def __call__(self, logits, targets):
        import torch  # type: ignore

        class_counts = self.class_count_tensor.to(logits.device, dtype=logits.dtype)
        shifted_logits = logits + torch.log(class_counts).unsqueeze(0)
        return torch.nn.functional.cross_entropy(shifted_logits, targets)


class LDAMDRWLoss:
    def __init__(
        self,
        class_counts: dict[str, int],
        class_names: list[str] | tuple[str, ...],
        *,
        max_m: float = 0.5,
        scale: float = 30.0,
        drw_start_epoch: int = 8,
    ) -> None:
        import torch  # type: ignore

        counts = torch.tensor([max(1, int(class_counts.get(class_name, 0))) for class_name in class_names], dtype=torch.float32)
        margins = 1.0 / torch.sqrt(torch.sqrt(counts))
        self.class_margins = margins * (float(max_m) / float(margins.max().item()))
        self.class_weights = compute_class_weights(class_names, class_counts, normalize=True)
        self.scale = float(scale)
        self.drw_start_epoch = int(drw_start_epoch)
        self.current_weight = None

    def on_epoch_start(self, epoch: int) -> None:
        self.current_weight = self.class_weights if int(epoch) >= self.drw_start_epoch else None

    def __call__(self, logits, targets):
        import torch  # type: ignore

        margins = self.class_margins.to(logits.device, dtype=logits.dtype)
        margin_per_sample = margins.gather(0, targets)
        adjusted_logits = logits.clone()
        adjusted_logits[torch.arange(logits.size(0), device=logits.device), targets] -= margin_per_sample
        weights = None if self.current_weight is None else self.current_weight.to(logits.device, dtype=logits.dtype)
        return torch.nn.functional.cross_entropy(self.scale * adjusted_logits, targets, weight=weights)


class StructuredTGLoss:
    needs_batch = True

    def __init__(
        self,
        *,
        class_counts: dict[str, int],
        class_names: list[str] | tuple[str, ...],
        t3_loss_name: str = "balanced_softmax",
        t1_weight: float = 1.0,
        t2_weight: float = 1.0,
        t3_weight: float = 1.0,
        leaf_weight: float = 0.0,
        label_smoothing: float = 0.0,
        ldam_max_m: float = 0.5,
        ldam_scale: float = 30.0,
        drw_start_epoch: int = 8,
    ) -> None:
        import torch  # type: ignore

        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        required = ("no_ulcer", "micro_punctate", "macro_punctate", "coalescent_macro_punctate", "patch_gt_1mm")
        missing = [name for name in required if name not in self.class_to_index]
        if missing:
            raise ValueError(f"StructuredTGLoss requires TG class names, missing: {missing}")

        self.no_ulcer_index = self.class_to_index["no_ulcer"]
        self.patch_index = self.class_to_index["patch_gt_1mm"]
        self.punctate_class_names = ("micro_punctate", "macro_punctate", "coalescent_macro_punctate")
        self.punctate_indices = [self.class_to_index[name] for name in self.punctate_class_names]

        self.t1_loss = torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
        self.t2_loss = torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
        self.leaf_loss = torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

        punctate_counts = {name: int(class_counts.get(name, 0)) for name in self.punctate_class_names}
        t3_name = t3_loss_name.strip().lower()
        if t3_name in {"balanced_softmax", "balancedsoftmax"}:
            self.t3_loss = BalancedSoftmaxLoss(class_counts=punctate_counts, class_names=self.punctate_class_names)
        elif t3_name in {"ldam_drw", "ldam"}:
            self.t3_loss = LDAMDRWLoss(
                class_counts=punctate_counts,
                class_names=self.punctate_class_names,
                max_m=ldam_max_m,
                scale=ldam_scale,
                drw_start_epoch=drw_start_epoch,
            )
        elif t3_name in {"weighted_ce", "weighted_cross_entropy"}:
            weights = compute_class_weights(self.punctate_class_names, punctate_counts, normalize=True)
            self.t3_loss = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=float(label_smoothing))
        elif t3_name in {"ce", "cross_entropy"}:
            self.t3_loss = torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
        else:
            raise ValueError(f"Unsupported T3 loss for structured TG: {t3_loss_name}")

        self.t1_weight = float(t1_weight)
        self.t2_weight = float(t2_weight)
        self.t3_weight = float(t3_weight)
        self.leaf_weight = float(leaf_weight)

    def on_epoch_start(self, epoch: int) -> None:
        if hasattr(self.t3_loss, "on_epoch_start"):
            self.t3_loss.on_epoch_start(epoch)

    def __call__(self, outputs, batch):
        import torch  # type: ignore

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        targets = batch["target"]

        t1_logits = torch.stack(
            [
                logits[:, self.no_ulcer_index],
                torch.logsumexp(
                    torch.cat(
                        [
                            logits[:, : self.no_ulcer_index],
                            logits[:, self.no_ulcer_index + 1 :],
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
            ],
            dim=1,
        )
        t1_targets = (targets != self.no_ulcer_index).long()
        t1_loss = self.t1_loss(t1_logits, t1_targets)

        ulcer_mask = targets != self.no_ulcer_index
        if ulcer_mask.any():
            ulcer_logits = logits[ulcer_mask]
            t2_logits = torch.stack(
                [
                    torch.logsumexp(ulcer_logits[:, self.punctate_indices], dim=1),
                    ulcer_logits[:, self.patch_index],
                ],
                dim=1,
            )
            t2_targets = (targets[ulcer_mask] == self.patch_index).long()
            t2_loss = self.t2_loss(t2_logits, t2_targets)
        else:
            t2_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        punctate_mask = torch.zeros_like(targets, dtype=torch.bool)
        for punctate_index in self.punctate_indices:
            punctate_mask |= targets == punctate_index
        if punctate_mask.any():
            punctate_logits = logits[punctate_mask][:, self.punctate_indices]
            punctate_targets = targets[punctate_mask].clone()
            for relative_index, class_index in enumerate(self.punctate_indices):
                punctate_targets[punctate_targets == class_index] = relative_index
            t3_loss = self.t3_loss(punctate_logits, punctate_targets)
        else:
            t3_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        if self.leaf_weight > 0.0:
            leaf_loss = self.leaf_loss(logits, targets)
        else:
            leaf_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        total_loss = (
            (self.t1_weight * t1_loss)
            + (self.t2_weight * t2_loss)
            + (self.t3_weight * t3_loss)
            + (self.leaf_weight * leaf_loss)
        )
        return total_loss, {
            "t1_loss": float(t1_loss.detach().item()),
            "t2_loss": float(t2_loss.detach().item()),
            "t3_loss": float(t3_loss.detach().item()),
            "leaf_loss": float(leaf_loss.detach().item()),
        }


def build_loss(
    loss_name: str,
    class_weights=None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    class_counts: dict[str, int] | None = None,
    class_names: list[str] | tuple[str, ...] | None = None,
    ldam_max_m: float = 0.5,
    ldam_scale: float = 30.0,
    drw_start_epoch: int = 8,
    tg_t3_loss_name: str = "balanced_softmax",
    tg_t1_weight: float = 1.0,
    tg_t2_weight: float = 1.0,
    tg_t3_weight: float = 1.0,
    tg_leaf_weight: float = 0.0,
):
    import torch  # type: ignore

    name = loss_name.lower()
    if name in {"ce", "cross_entropy"}:
        return torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    if name in {"weighted_ce", "weighted_cross_entropy"}:
        return torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(label_smoothing))
    if name == "focal":
        return FocalLoss(weight=class_weights, gamma=focal_gamma)
    if name in {"balanced_softmax", "balancedsoftmax"}:
        if class_counts is None or class_names is None:
            raise ValueError("Balanced Softmax requires class_counts and class_names.")
        return BalancedSoftmaxLoss(class_counts=class_counts, class_names=class_names)
    if name in {"ldam_drw", "ldam"}:
        if class_counts is None or class_names is None:
            raise ValueError("LDAM-DRW requires class_counts and class_names.")
        return LDAMDRWLoss(
            class_counts=class_counts,
            class_names=class_names,
            max_m=ldam_max_m,
            scale=ldam_scale,
            drw_start_epoch=drw_start_epoch,
        )
    if name in {"tg_structured_t123", "structured_tg_t123"}:
        if class_counts is None or class_names is None:
            raise ValueError("Structured TG loss requires class_counts and class_names.")
        return StructuredTGLoss(
            class_counts=class_counts,
            class_names=class_names,
            t3_loss_name=tg_t3_loss_name,
            t1_weight=tg_t1_weight,
            t2_weight=tg_t2_weight,
            t3_weight=tg_t3_weight,
            leaf_weight=tg_leaf_weight,
            label_smoothing=label_smoothing,
            ldam_max_m=ldam_max_m,
            ldam_scale=ldam_scale,
            drw_start_epoch=drw_start_epoch,
        )
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
