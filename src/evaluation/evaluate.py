from __future__ import annotations

from typing import Any


def _build_progress(iterable, *, desc: str | None, enabled: bool, leave: bool):
    if not enabled or not desc:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(iterable, desc=desc, total=len(iterable), leave=leave, dynamic_ncols=True, colour="cyan")
    except Exception:
        return iterable


def run_inference(model, dataloader, device: str, criterion=None, progress_desc: str | None = None, show_progress: bool = False):
    import numpy as np  # type: ignore
    import torch  # type: ignore

    model.eval()
    losses: list[float] = []
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probabilities: list[np.ndarray] = []
    prediction_rows: list[dict[str, Any]] = []

    iterator = _build_progress(dataloader, desc=progress_desc, enabled=show_progress, leave=False)
    try:
        with torch.inference_mode():
            for batch in iterator:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                logits = model(images)
                if criterion is not None:
                    losses.append(float(criterion(logits, targets).item()))
                probabilities = torch.softmax(logits, dim=1)
                preds = probabilities.argmax(dim=1)

                all_targets.extend(targets.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_probabilities.extend(probabilities.cpu().numpy())

                target_list = targets.cpu().tolist()
                pred_list = preds.cpu().tolist()
                for row_index, image_id in enumerate(batch["image_id"]):
                    row = {
                        "image_id": str(image_id),
                        "split": str(batch["split"][row_index]),
                        "target_index": int(target_list[row_index]),
                        "predicted_index": int(pred_list[row_index]),
                        "confidence": float(probabilities[row_index].max().item()),
                        "raw_image_path": str(batch["raw_image_path"][row_index]),
                        "cornea_mask_path": str(batch["cornea_mask_path"][row_index]) if "cornea_mask_path" in batch else "",
                        "ulcer_mask_path": str(batch["ulcer_mask_path"][row_index]) if "ulcer_mask_path" in batch else "",
                    }
                    prediction_rows.append(row)
    finally:
        close_method = getattr(iterator, "close", None)
        if callable(close_method):
            close_method()

    mean_loss = float(sum(losses) / max(1, len(losses))) if losses else None
    probabilities_array = np.asarray(all_probabilities) if all_probabilities else None
    return {
        "loss": mean_loss,
        "y_true": all_targets,
        "y_pred": all_preds,
        "probabilities": probabilities_array,
        "prediction_rows": prediction_rows,
    }
