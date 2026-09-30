"""Metrics use deceptive as the positive class, despite its numeric code of zero."""

import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from review_authenticity.labels import LABEL_NAMES, LABEL_ORDER, Label

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def measure(y_true, genuine_probability, threshold: float) -> dict:
    """Compute classification and probability quality from actual predictions."""
    p = np.asarray(genuine_probability, dtype=float)
    predicted = np.where(p >= threshold, int(Label.GENUINE), int(Label.DECEPTIVE))
    matrix = confusion_matrix(y_true, predicted, labels=LABEL_ORDER)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(
            precision_score(y_true, predicted, pos_label=Label.DECEPTIVE, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predicted, pos_label=Label.DECEPTIVE, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predicted, pos_label=Label.DECEPTIVE, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, p, pos_label=Label.GENUINE)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - p, p]), labels=LABEL_ORDER)),
        "confusion_matrix": matrix.tolist(),
        "label_order": [LABEL_NAMES[label] for label in LABEL_ORDER],
        "positive_class": "deceptive",
        "false_positives": int(matrix[1, 0]),
        "false_negatives": int(matrix[0, 1]),
        "sample_count": len(p),
        "classification_report": classification_report(
            y_true,
            predicted,
            labels=LABEL_ORDER,
            target_names=[LABEL_NAMES[label] for label in LABEL_ORDER],
            zero_division=0,
        ),
    }


def write_json(path: Path, value: dict) -> None:
    """Write portable metadata without allowing non-finite numeric values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_report(directory: Path, metrics: dict) -> None:
    """Save JSON, a readable classification report, and a confusion matrix plot."""
    write_json(directory / "metrics.json", metrics)
    (directory / "classification_report.txt").write_text(
        metrics["classification_report"], encoding="utf-8"
    )
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    display = ConfusionMatrixDisplay(
        np.asarray(metrics["confusion_matrix"]), display_labels=metrics["label_order"]
    )
    display.plot(cmap="Blues", colorbar=False)
    display.ax_.set_title("Held-out test set" if metrics.get("split") == "test" else "Evaluation")
    display.figure_.tight_layout()
    display.figure_.savefig(figures / "confusion_matrix.png", dpi=150)
    plt.close(display.figure_)
