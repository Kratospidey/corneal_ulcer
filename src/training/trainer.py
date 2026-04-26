from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from console_utils import emit_epoch_summary
from evaluation.evaluate import run_inference
from evaluation.metrics import compute_classification_metrics
from model_factory import model_outputs_to_dict, primary_logits
from training.losses import compute_distillation_kl, compute_ordinal_aux_loss
from training.optim_utils import step_scheduler


@dataclass
class TrainingResult:
    history: list[dict[str, Any]] = field(default_factory=list)
    best_metric: float = float("-inf")
    best_epoch: int = -1
    checkpoint_path: str = ""


def fit_model(
    model,
    loaders,
    criterion,
    optimizer,
    scheduler,
    class_names: list[str] | tuple[str, ...],
    device: str,
    training_config: dict[str, Any],
    checkpoint_path,
    console=None,
    teacher_model=None,
    distillation_config: dict[str, Any] | None = None,
):
    import torch  # type: ignore

    epochs = int(training_config.get("epochs", 10))
    patience = int(training_config.get("early_stopping_patience", 4))
    grad_clip_norm = float(training_config.get("grad_clip_norm", 1.0))
    use_amp = bool(training_config.get("amp", True) and device == "cuda")
    show_progress = bool(training_config.get("show_progress", True))
    best_metric_name = str(training_config.get("best_metric", "balanced_accuracy"))
    ordinal_aux_weight = float(training_config.get("ordinal_aux_weight", 0.0))
    distillation_config = dict(distillation_config or {})
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    result = TrainingResult(checkpoint_path=str(checkpoint_path))
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
            num_classes=len(class_names),
            ordinal_aux_weight=ordinal_aux_weight,
            teacher_model=teacher_model,
            distillation_config=distillation_config,
            progress_desc=f"Train {epoch}/{epochs}",
            show_progress=show_progress,
        )
        val_payload = run_inference(
            model,
            loaders["val"],
            device=device,
            criterion=criterion,
            progress_desc=f"Val {epoch}/{epochs}",
            show_progress=show_progress,
        )
        val_metrics_payload = compute_classification_metrics(
            val_payload["y_true"],
            val_payload["y_pred"],
            val_payload["probabilities"],
            class_names=class_names,
        )
        val_metric = float(val_metrics_payload["metrics"][best_metric_name])
        result.history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_payload["loss"],
                "val_balanced_accuracy": val_metrics_payload["metrics"]["balanced_accuracy"],
                "val_macro_f1": val_metrics_payload["metrics"]["macro_f1"],
                "lr": train_metrics["lr"],
                "train_cls_loss": train_metrics["cls_loss"],
                "train_ord_loss": train_metrics["ord_loss"],
                "train_distill_loss": train_metrics["distill_loss"],
            }
        )

        improved = False
        if val_metric > result.best_metric:
            result.best_metric = val_metric
            result.best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            improved = True
        else:
            stale_epochs += 1

        emit_epoch_summary(
            console,
            epoch=epoch,
            epochs=epochs,
            train_loss=train_metrics["loss"],
            val_loss=val_payload["loss"],
            val_balanced_accuracy=val_metrics_payload["metrics"]["balanced_accuracy"],
            val_macro_f1=val_metrics_payload["metrics"]["macro_f1"],
            best_metric_name=best_metric_name,
            best_metric=result.best_metric,
            best_epoch=result.best_epoch,
            improved=improved,
        )
        step_scheduler(scheduler, monitor_value=val_metric)
        if stale_epochs >= patience:
            break
    return result


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device: str,
    use_amp: bool,
    scaler,
    grad_clip_norm: float,
    num_classes: int,
    ordinal_aux_weight: float,
    teacher_model=None,
    distillation_config: dict[str, Any] | None = None,
    progress_desc: str | None = None,
    show_progress: bool = False,
):
    import torch  # type: ignore

    model.train()
    losses: list[float] = []
    cls_losses: list[float] = []
    ord_losses: list[float] = []
    distill_losses: list[float] = []
    distillation_config = dict(distillation_config or {})
    distill_enabled = bool(distillation_config.get("enabled", False) and teacher_model is not None)
    distill_weight = float(distillation_config.get("weight", 0.0))
    distill_temperature = float(distillation_config.get("temperature", 2.0))
    if distill_enabled:
        teacher_model.eval()
    iterator = dataloader
    if show_progress and progress_desc:
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(dataloader, desc=progress_desc, total=len(dataloader), leave=False, dynamic_ncols=True, colour="green")
        except Exception:
            iterator = dataloader
    try:
        for batch in iterator:
            if "image_tight" in batch and "image_wide" in batch:
                images = (batch["image_tight"].to(device, non_blocking=True), batch["image_wide"].to(device, non_blocking=True))
            else:
                images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                model_outputs = model_outputs_to_dict(model(images))
                logits = primary_logits(model_outputs)
                cls_loss = criterion(logits, targets)
                ord_loss = torch.zeros((), device=targets.device, dtype=cls_loss.dtype)
                if ordinal_aux_weight > 0.0 and model_outputs.get("ordinal_logits") is not None:
                    ord_loss = compute_ordinal_aux_loss(
                        model_outputs["ordinal_logits"],
                        targets,
                        num_classes=num_classes,
                    )
                distill_loss = torch.zeros((), device=targets.device, dtype=cls_loss.dtype)
                if distill_enabled and distill_weight > 0.0:
                    with torch.no_grad():
                        teacher_outputs = model_outputs_to_dict(teacher_model(images))
                        teacher_logits = primary_logits(teacher_outputs)
                    distill_loss = compute_distillation_kl(
                        logits,
                        teacher_logits,
                        temperature=distill_temperature,
                    )
                loss = cls_loss + (float(ordinal_aux_weight) * ord_loss) + (distill_weight * distill_loss)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            losses.append(float(loss.item()))
            cls_losses.append(float(cls_loss.item()))
            ord_losses.append(float(ord_loss.item()))
            distill_losses.append(float(distill_loss.item()))
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    loss=f"{losses[-1]:.4f}",
                    cls=f"{cls_losses[-1]:.4f}",
                    ord=f"{ord_losses[-1]:.4f}",
                    dst=f"{distill_losses[-1]:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
    finally:
        close_method = getattr(iterator, "close", None)
        if callable(close_method):
            close_method()
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "cls_loss": float(sum(cls_losses) / max(1, len(cls_losses))),
        "ord_loss": float(sum(ord_losses) / max(1, len(ord_losses))),
        "distill_loss": float(sum(distill_losses) / max(1, len(distill_losses))),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def save_checkpoint(model, optimizer, epoch: int, checkpoint_path) -> None:
    import torch  # type: ignore

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
