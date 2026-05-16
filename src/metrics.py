"""评估指标: Accuracy, Macro-F1, per-class recall, confusion matrix。"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix as sklearn_cm,
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(all_labels, all_preds, class_names):
    """计算全部评估指标。

    Returns:
        metrics: dict，包含 accuracy, macro_f1, recall_{cls}
        conf_matrix: np.ndarray，混淆矩阵
    """
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    conf_matrix = sklearn_cm(all_labels, all_preds)

    metrics = {
        "accuracy": acc * 100,
        "macro_f1": macro_f1 * 100,
    }
    for i, name in enumerate(class_names):
        metrics[f"recall_{name}"] = per_class_recall[i] * 100

    return metrics, conf_matrix


def plot_confusion_matrix(conf_matrix, class_names, save_path):
    """绘制并保存混淆矩阵。"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
