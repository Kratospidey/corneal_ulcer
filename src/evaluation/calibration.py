from __future__ import annotations


def compute_calibration(probabilities, y_true):
    import numpy as np  # type: ignore

    if probabilities is None:
        return {"ece": None, "mean_brier_like": None}

    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences >= lower) & (confidences < upper if upper < 1.0 else confidences <= upper)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = correctness[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(avg_conf - avg_acc)

    target_probs = probabilities[np.arange(len(y_true)), y_true]
    brier_like = ((1.0 - target_probs) ** 2).mean()
    return {"ece": float(ece), "mean_brier_like": float(brier_like)}
