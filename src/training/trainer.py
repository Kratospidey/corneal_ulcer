from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.evaluate import run_inference
from evaluation.metrics import compute_classification_metrics
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
):
    import torch  # type: ignore

    epochs = int(training_config.get("epochs", 10))
    patience = int(training_config.get("early_stopping_patience", 4))
    grad_clip_norm = float(training_config.get("grad_clip_norm", 1.0))
    use_amp = bool(training_config.get("amp", True) and device == "cuda")
    best_metric_name = str(training_config.get("best_metric", "balanced_accuracy"))
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
        )
        val_payload = run_inference(model, loaders["val"], device=device, criterion=criterion)
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
            }
        )

        if val_metric > result.best_metric:
            result.best_metric = val_metric
            result.best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
        else:
            stale_epochs += 1

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
):
    import torch  # type: ignore

    model.train()
    losses: list[float] = []
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
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
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
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
