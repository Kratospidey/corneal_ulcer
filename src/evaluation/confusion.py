from __future__ import annotations

from pathlib import Path


def save_confusion_matrix(y_true, y_pred, class_names: list[str] | tuple[str, ...], output_csv: str | Path, output_png: str | Path) -> None:
    from sklearn.metrics import confusion_matrix  # type: ignore
    import matplotlib  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    matplotlib.use("Agg")
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(output_csv)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 6))
    im = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=35, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=axis)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
