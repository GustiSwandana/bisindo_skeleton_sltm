from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use('Agg')


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_index, pred_index in zip(y_true, y_pred):
        matrix[int(true_index), int(pred_index)] += 1
    return matrix


def compute_classification_report(
    confusion_matrix: np.ndarray,
    class_names: list[str],
) -> dict:
    rows = []
    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for index, class_name in enumerate(class_names):
        true_positive = int(confusion_matrix[index, index])
        false_positive = int(np.sum(confusion_matrix[:, index]) - true_positive)
        false_negative = int(np.sum(confusion_matrix[index, :]) - true_positive)
        support = int(np.sum(confusion_matrix[index, :]))

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        rows.append(
            {
                'class_name': class_name,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': support,
            }
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        supports.append(support)

    total_support = int(sum(supports))
    correct_predictions = int(np.trace(confusion_matrix))
    accuracy = correct_predictions / total_support if total_support else 0.0
    macro_avg = {
        'precision': float(np.mean(precisions)) if precisions else 0.0,
        'recall': float(np.mean(recalls)) if recalls else 0.0,
        'f1_score': float(np.mean(f1_scores)) if f1_scores else 0.0,
        'support': total_support,
    }
    weighted_avg = {
        'precision': float(np.average(precisions, weights=supports)) if total_support else 0.0,
        'recall': float(np.average(recalls, weights=supports)) if total_support else 0.0,
        'f1_score': float(np.average(f1_scores, weights=supports)) if total_support else 0.0,
        'support': total_support,
    }

    return {
        'per_class': rows,
        'accuracy': accuracy,
        'macro_avg': macro_avg,
        'weighted_avg': weighted_avg,
    }


def save_confusion_matrix_figure(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_width = max(10, len(class_names) * 0.45)
    figure_height = max(8, len(class_names) * 0.45)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    image = ax.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel='Predicted Label',
        ylabel='True Label',
        title='Confusion Matrix BISINDO',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    threshold = confusion_matrix.max() / 2.0 if confusion_matrix.size else 0.0
    for row in range(confusion_matrix.shape[0]):
        for col in range(confusion_matrix.shape[1]):
            value = int(confusion_matrix[row, col])
            ax.text(
                col,
                row,
                f'{value}',
                ha='center',
                va='center',
                color='white' if value > threshold else 'black',
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
