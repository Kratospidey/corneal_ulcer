from __future__ import annotations

from typing import Any


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    probabilities,
    class_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        f1_score,
        precision_recall_curve,
        average_precision_score,
        roc_auc_score,
        roc_curve,
    )  # type: ignore
    from sklearn.preprocessing import label_binarize  # type: ignore
    import numpy as np  # type: ignore

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report

    curves: dict[str, Any] = {"roc": {}, "pr": {}}
    if probabilities is None:
        metrics["roc_auc_macro_ovr"] = None
        metrics["pr_auc_macro_ovr"] = None
        return {"metrics": metrics, "curves": curves}

    probabilities = np.asarray(probabilities)
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    try:
        metrics["roc_auc_macro_ovr"] = float(
            roc_auc_score(y_true_bin, probabilities, average="macro", multi_class="ovr")
        )
    except ValueError:
        metrics["roc_auc_macro_ovr"] = None
    try:
        metrics["pr_auc_macro_ovr"] = float(average_precision_score(y_true_bin, probabilities, average="macro"))
    except ValueError:
        metrics["pr_auc_macro_ovr"] = None

    for class_index, class_name in enumerate(class_names):
        y_true_class = y_true_bin[:, class_index]
        y_prob_class = probabilities[:, class_index]
        if len(set(y_true_class.tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_class, y_prob_class)
        precision, recall, _ = precision_recall_curve(y_true_class, y_prob_class)
        curves["roc"][class_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        curves["pr"][class_name] = {"precision": precision.tolist(), "recall": recall.tolist()}
    return {"metrics": metrics, "curves": curves}


def flatten_classification_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, payload in report.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "label": label,
                "precision": payload.get("precision"),
                "recall": payload.get("recall"),
                "f1_score": payload.get("f1-score"),
                "support": payload.get("support"),
            }
        )
    return rows
