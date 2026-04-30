"""Evaluation utilities: curves, confusion matrix, and reports."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    CLASSIFICATION_REPORT_PATH,
    CLASS_NAMES,
    CONFUSION_MATRIX_PATH,
    OUTPUT_DIR,
    TRAINING_CURVES_PATH,
)


def plot_training_curves(history: dict[str, list[float]]) -> None:
    """Plot and save accuracy/loss curves in one figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("accuracy", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history.get("accuracy", []), label="Train Accuracy")
    axes[0].plot(epochs, history.get("val_accuracy", []), label="Val Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history.get("loss", []), label="Train Loss")
    axes[1].plot(epochs, history.get("val_loss", []), label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(TRAINING_CURVES_PATH, dpi=200)
    plt.close(fig)


def evaluate_and_report(model, x_test: np.ndarray, y_test: np.ndarray) -> str:
    """Generate confusion matrix and classification report, then save artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("CIFAR-10 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()

    with CLASSIFICATION_REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write(report)

    return report

