"""Small plotting helpers for saved epoch histories and test confusion matrices."""
from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(history, output):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split, style in (("train", "-"), ("val", "--")):
        axes[0].plot(epochs, [row[split]["loss"] for row in history], style, label=split)
        axes[1].plot(epochs, [row[split]["accuracy"] for row in history], style, label=split)
    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(matrix, class_names, output):
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axis.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
