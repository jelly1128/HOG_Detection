from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
import numpy as np
from labeling.label_converter import HardMultiLabelResult
from labeling.result_types import SingleLabelResult
import pandas as pd
from pathlib import Path

class ClassificationMetricsCalculator:
    def __init__(self, num_classes: int = 15, mode: str = "multitask"):
        """
        Args:
            num_classes (int or None): クラス数。Noneの場合はmodeに応じて自動設定。
            mode (str): "multitask"（マルチラベル/マルチタスク）または "single_label"（シングルラベル）
        """
        self.num_classes = num_classes
        self.mode = mode
        
    def calculate_metrics_multi_label_per_class(self, y_true, y_pred):
        """
        各クラスごとのTP, FP, TN, FNを計算
        
        Args:
            y_true: one-hot形式の正解ラベル (n_samples, n_classes)
            y_pred: one-hot形式の予測ラベル (n_samples, n_classes)
            
        Returns:
            precision: 各クラスの適合率
            recall: 各クラスの再現率
            f1_score: 各クラスのF1スコア
            accuracies: 各クラスの正解率
            confusion_matrices: 各クラスの混同行列
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n_classes = y_true.shape[1]
        
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []
        confusion_matrices = []
        
        for class_idx in range(n_classes):
            tp = np.sum((y_true[:, class_idx] == 1) & (y_pred[:, class_idx] == 1))
            fp = np.sum((y_true[:, class_idx] == 0) & (y_pred[:, class_idx] == 1))
            tn = np.sum((y_true[:, class_idx] == 0) & (y_pred[:, class_idx] == 0))
            fn = np.sum((y_true[:, class_idx] == 1) & (y_pred[:, class_idx] == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            accuracies.append(accuracy)
            confusion_matrices.append([[tp, fp], [fn, tn]])
        
        return np.array(precisions), np.array(recalls), np.array(f1_scores), np.array(accuracies), np.array(confusion_matrices)
    
    
    def calculate_metrics_single_label_per_class(self, y_true, y_pred):
        """
        シングルラベル（0-5のクラスラベル）の予測に対して、
        各クラスごとのTP, FP, TN, FNを計算

        Args:
            y_true: 0-5のクラスラベルの正解ラベル (n_samples,)
            y_pred: 0-5のクラスラベルの予測ラベル (n_samples,)

        Returns:
            precision: 各クラスの適合率
            recall: 各クラスの再現率
            accuracy: 各クラスの正解率
            confusion_matrices: 各クラスの混同行列
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # クラス数×クラス数の混同行列を計算
        # modeに応じてクラス数を設定
        if self.mode == "multitask":
            n_classes = 6  # 主クラスの数
        elif self.mode == "single_label":
            n_classes = 2
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        # 混同行列を作成
        for t, p in zip(y_true, y_pred):
            confusion_matrix[t, p] += 1
        
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []
        confusion_matrices = []
        
        for class_idx in range(n_classes):
            tp = confusion_matrix[class_idx, class_idx]
            fp = np.sum(confusion_matrix[:, class_idx]) - tp
            fn = np.sum(confusion_matrix[class_idx, :]) - tp
            tn = np.sum(confusion_matrix) - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            accuracies.append(accuracy)
            confusion_matrices.append([[tp, fp], [fn, tn]])
            
        return np.array(precisions), np.array(recalls), np.array(f1_scores), np.array(accuracies), np.array(confusion_matrices)
            

    def calculate_multi_label_metrics_per_video(self, hard_multi_label_results: dict[str, HardMultiLabelResult]) -> dict[str, dict[str, float]]:
        """
        各動画単位の混同行列と適合率・再現率・正解率を計算する関数

        Args:
            hard_multi_label_results (dict[str, HardMultiLabelResult]): 各フォルダのマルチラベルの結果

        Returns:
            dict: 各動画のメトリクスを格納した辞書
                [folder_name] = {
                    'precision': 適合率,
                    'recall': 再現率,
                    'f1_score': F1スコア,
                    'accuracy': 正解率,
                    'confusion_matrix': 混同行列
                }
        """
        video_metrics = {}

        for folder_name, hard_multi_label_result in hard_multi_label_results.items():
            y_true = np.array(hard_multi_label_result.ground_truth_labels)
            y_pred = np.array(hard_multi_label_result.multi_labels)
            
            precision, recall, f1_score, accuracy, cm = self.calculate_metrics_multi_label_per_class(y_true, y_pred)
            video_metrics[folder_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'confusion_matrix': cm
            }
        
        return video_metrics

    def calculate_multi_label_overall_metrics(self, hard_multi_label_results: dict[str, HardMultiLabelResult]):
        """
        全動画の混同行列と適合率・再現率・正解率を計算する関数

        Args:
            hard_multi_label_results (dict[str, HardMultiLabelResult]): 各フォルダのマルチラベルの結果

        Returns:
            dict: 全動画のメトリクスを格納した辞書
                - class_metrics: 各クラスの適合率・再現率・正解率
                - per_class_confusion_matrices: 各クラスの2×2混同行列
                - class_confusion_matrix: クラス数×クラス数の混同行列
        """
        y_true = []
        y_pred = []

        for hard_multi_label_result in hard_multi_label_results.values():
            y_true.extend(hard_multi_label_result.ground_truth_labels)
            y_pred.extend(hard_multi_label_result.multi_labels)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # クラスごとの混同行列を計算
        per_class_confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
        
        # クラス数×クラス数の混同行列を計算
        n_classes = y_true.shape[1]
        class_confusion_matrix = np.zeros((n_classes, n_classes))
        
        # 各サンプルについて、予測クラスと真のクラスの組み合わせをカウント
        for true_labels, pred_labels in zip(y_true, y_pred):
            true_indices = np.where(true_labels == 1)[0]
            pred_indices = np.where(pred_labels == 1)[0]
            
            # 真のクラスと予測クラスの組み合わせをカウント
            for true_idx in true_indices:
                for pred_idx in pred_indices:
                    class_confusion_matrix[true_idx, pred_idx] += 1
        
        # 各クラスの精度指標を計算
        class_metrics = []
        for class_idx in range(n_classes):
            cm = per_class_confusion_matrices[class_idx]
            tp, fp = cm[1, 1], cm[0, 1]
            fn, tn = cm[1, 0], cm[0, 0]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            
            class_metrics.append({
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy
            })
    
        return {
            'class_metrics': class_metrics,  # 各クラスの適合率・再現率・F1スコア・正解率
            'per_class_confusion_matrices': per_class_confusion_matrices,  # 各クラスの2×2混同行列
            'class_confusion_matrix': class_confusion_matrix  # クラス数×クラス数の混同行列
        }
        
    def calculate_single_label_metrics_per_video(self, single_label_results: dict[str, SingleLabelResult]) -> dict[str, dict[str, float]]:
        """
        各動画単位の混同行列と適合率・再現率・正解率を計算する関数

        Args:
            single_label_results (dict[str, SingleLabelResult]): 各フォルダのシングルラベルの結果
        Returns:
            dict: 各動画のメトリクスを格納した辞書
        """
        video_metrics = {}

        for folder_name, single_label_result in single_label_results.items():
            y_true = np.array(single_label_result.ground_truth_labels)
            y_pred = np.array(single_label_result.single_labels)
            
            precision, recall, f1_score, accuracy, cm = self.calculate_metrics_single_label_per_class(y_true, y_pred)
            video_metrics[folder_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'confusion_matrix': cm
            }
        
        return video_metrics
    
    def calculate_single_label_overall_metrics(self, single_label_results: dict[str, SingleLabelResult]):
        """
        シングルラベル（0-5のクラスラベル）の予測に対して、
        全動画の混同行列と各クラスの精度指標を計算する関数

        Args:
            single_label_results (dict[str, SingleLabelResult]): 各フォルダのシングルラベルの結果

        Returns:
            dict: 全動画のメトリクスを格納した辞書
                - class_metrics: 各クラスの適合率・再現率・正解率
                - per_class_confusion_matrices: 各クラスの2×2混同行列
                - class_confusion_matrix: クラス数×クラス数の混同行列
        """
        y_true = []
        y_pred = []

        for single_label_result in single_label_results.values():
            y_true.extend(single_label_result.ground_truth_labels)
            y_pred.extend(single_label_result.single_labels)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # クラス数×クラス数の混同行列を計算
        # modeに応じてクラス数を設定
        if self.mode == "multitask":
            n_classes = 6  # 主クラスの数
        elif self.mode == "single_label":
            n_classes = 2
        class_confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        # 混同行列を作成
        for t, p in zip(y_true, y_pred):
            class_confusion_matrix[t, p] += 1
        
        # 各クラスの2×2混同行列と精度指標を計算
        per_class_confusion_matrices = []
        class_metrics = []
        
        for class_idx in range(n_classes):
            tp = class_confusion_matrix[class_idx, class_idx]
            fp = np.sum(class_confusion_matrix[:, class_idx]) - tp
            fn = np.sum(class_confusion_matrix[class_idx, :]) - tp
            tn = np.sum(class_confusion_matrix) - tp - fp - fn
            
            per_class_confusion_matrices.append([[tn, fp], [fn, tp]])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            
            class_metrics.append({
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy
            })
        
        return {
            'class_metrics': class_metrics,  # 各クラスの適合率・再現率・F1スコア・正解率
            'per_class_confusion_matrices': np.array(per_class_confusion_matrices),  # 各クラスの2×2混同行列
            'class_confusion_matrix': class_confusion_matrix  # クラス数×クラス数の混同行列
        }
            
    def calculate_all_folds_metrics(self, all_folds_results: dict[str, SingleLabelResult], save_dir: Path):
        """
        全フォールドのスライディングウィンドウ適用後の結果を統合して評価指標を計算し保存します。
        
        Args:
            all_folds_results (dict[str, SingleLabelResult]): 全フォールドの結果を含む辞書
            save_dir (Path): 結果を保存するディレクトリのパス

        Returns:
            dict: 全フォールドのメトリクスを格納した辞書
            - class_metrics: 各クラスの適合率・再現率・F1スコア・正解率
            - per_class_confusion_matrices: 各クラスの2×2混同行列
        """
        y_true = []
        y_pred = []
        
        # 全フォールドのデータを統合
        for single_label_result in all_folds_results.values():
            y_pred.extend(single_label_result.single_labels)
            y_true.extend(single_label_result.ground_truth_labels)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # クラス数×クラス数の混同行列を計算
        # modeに応じてクラス数を設定
        if self.mode == "multitask":
            n_classes = 6  # 主クラスの数
        elif self.mode == "single_label":
            n_classes = 2
        class_confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        # 混同行列を作成
        for t, p in zip(y_true, y_pred):
            class_confusion_matrix[t, p] += 1
        
        # 各クラスの2×2混同行列と精度指標を計算
        per_class_confusion_matrices = []
        class_metrics = []
        
        for class_idx in range(n_classes):
            tp = class_confusion_matrix[class_idx, class_idx]
            fp = np.sum(class_confusion_matrix[:, class_idx]) - tp
            fn = np.sum(class_confusion_matrix[class_idx, :]) - tp
            tn = np.sum(class_confusion_matrix) - tp - fp - fn
            
            per_class_confusion_matrices.append([[tn, fp], [fn, tp]])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            
            class_metrics.append({
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy
            })
        
        # クラス数×クラス数の混同行列を保存
        cm_df = pd.DataFrame(
            class_confusion_matrix,
            index=[f'True_{i}' for i in range(n_classes)],
            columns=[f'Pred_{i}' for i in range(n_classes)]
        )
        cm_df.to_csv(save_dir / 'confusion_matrix.csv')
        
        # クラスごとの評価指標の保存
        metrics_df = pd.DataFrame([
            {
                'Class': i,
                'Precision': round(metrics['precision'], 4),
                'Recall': round(metrics['recall'], 4), 
                'F1 Score': round(metrics['f1_score'], 4),
                'Accuracy': round(metrics['accuracy'], 4)
            }
            for i, metrics in enumerate(class_metrics)
        ])
        metrics_df.to_csv(save_dir / 'class_metrics.csv', index=False)
        
        return {
            'class_metrics': class_metrics,
            'class_confusion_matrix': class_confusion_matrix,
        }


