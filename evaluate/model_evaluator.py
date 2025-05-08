from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc, accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class ModelEvaluator:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def plot_roc_curve(self, true_labels, predicted_probabilities, class_index):
        """ROC曲線をプロットし保存する"""
        false_positive_rates, true_positive_rates, thresholds = roc_curve(true_labels, predicted_probabilities)
        area_under_curve = auc(false_positive_rates, true_positive_rates)
        youden_index = true_positive_rates - false_positive_rates
        best_threshold_index = np.argmax(youden_index)
        best_threshold = thresholds[best_threshold_index]

        plt.plot(false_positive_rates, true_positive_rates, color='darkorange',
                 lw=2, label=f'ROC curve (AUC = {area_under_curve:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random Classifier')
        plt.scatter(false_positive_rates[best_threshold_index], true_positive_rates[best_threshold_index],
                    color='red', marker='o',
                    label=f'Best Threshold = {best_threshold:.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Class {class_index}')
        plt.legend(loc="lower right")

        plot_path = Path(self.save_dir, f'roc_curve_class_{class_index}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"ROC curve saved to {plot_path}")

    def evaluate_classification(self, y_true, y_pred):
        """分類評価指標を計算する"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return accuracy, precision, recall, f1

    def confusion_matrix(self, y_true, y_pred):
        """混同行列を計算し、表示する"""
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", cm)
        return cm
