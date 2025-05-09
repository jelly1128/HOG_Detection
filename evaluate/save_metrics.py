import csv
from pathlib import Path

def save_metrics_to_csv(video_metrics, overall_metrics, video_file_path, overall_file_path):
    """
    メトリクスをCSVファイルに保存する関数

    Args:
        video_metrics (dict): 各動画のメトリクス
        overall_metrics (dict): 全体のメトリクス
        video_file_path (str): 各動画のメトリクスを保存するCSVファイルのパス
        overall_file_path (str): 全体のメトリクスを保存するCSVファイルのパス
    """
    # 各動画のメトリクスをCSVに保存
    with open(video_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Video", "Class", "Precision", "Recall", "Accuracy"])
        
        for video, metrics in video_metrics.items():
            for class_idx, (precision, recall, accuracy) in enumerate(zip(metrics['precision'], metrics['recall'], metrics['accuracy'])):
                writer.writerow([video, class_idx, precision, recall, accuracy])
    
    # 全体のメトリクスをCSVに保存
    with open(overall_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Precision", "Recall", "Accuracy"])
        
        for class_idx, (precision, recall, accuracy) in enumerate(zip(overall_metrics['precision'], overall_metrics['recall'], overall_metrics['accuracy'])):
            writer.writerow([class_idx, precision, recall, accuracy])
        
        writer.writerow(["Overall", overall_metrics['overall_precision'], overall_metrics['overall_recall'], overall_metrics['overall_accuracy']])

def save_video_metrics_to_csv(video_metrics: dict[str, dict[str, float]], base_save_dir: Path, methods: str):
    """
    各動画フォルダにメトリクスをCSVファイルに保存する関数

    Args:
        video_metrics (dict): 各動画のメトリクス
        base_save_dir (Path): 保存するベースディレクトリのパス
    """
    for video_name, metrics in video_metrics.items():
        video_methods_results_dir = base_save_dir / video_name / methods
        video_methods_results_dir.mkdir(parents=True, exist_ok=True)

        # 混同行列を保存
        confusion_matrix_file = video_methods_results_dir / f'{methods}_confusion_matrix.csv'
        with open(confusion_matrix_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "TP", "FP", "FN", "TN"])
            for class_idx, cm in enumerate(metrics['confusion_matrix']):
                tp, fp, fn, tn = cm.ravel()
                writer.writerow([class_idx, tp, fp, fn, tn])

        # メトリクスを保存
        metrics_file = video_methods_results_dir / f'{methods}_metrics.csv'
        with open(metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "Precision", "Recall", "F1 score", "Accuracy"])
            for class_idx, (precision, recall, f1_score, accuracy) in enumerate(zip(metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['accuracy'])):
                writer.writerow([class_idx, 
                                 f"{precision:.4f}",
                                 f"{recall:.4f}",
                                 f"{f1_score:.4f}",
                                 f"{accuracy:.4f}"]
                                 )

def save_overall_metrics_to_csv(overall_metrics, base_save_dir: Path, methods: str):
    """
    全体のメトリクスをCSVファイルに保存する関数

    Args:
        overall_metrics (dict): 全体のメトリクス
        overall_file_path (str): 保存先のベースパス
        methods (str): 評価手法の名前
    """
    base_path = base_save_dir / methods
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 各クラスの精度指標を保存
    class_metrics_file = base_path / f'{methods}_class_metrics.csv'
    with open(class_metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Precision', 'Recall', "F1 score", 'Accuracy'])
        for class_idx, metrics in enumerate(overall_metrics['class_metrics']):
            writer.writerow([class_idx,
                             f"{metrics['precision']:.4f}",
                             f"{metrics['recall']:.4f}",
                             f"{metrics['f1_score']:.4f}",
                             f"{metrics['accuracy']:.4f}"
                            ])
    
    # 各クラスの2×2混同行列を保存
    per_class_cm_file = base_path / f'{methods}_per_class_confusion_matrices.csv'
    with open(per_class_cm_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'TP', 'FP', 'FN', 'TN'])
        for class_idx, cm in enumerate(overall_metrics['per_class_confusion_matrices']):
            tp, fp = cm[1, 1], cm[0, 1]
            fn, tn = cm[1, 0], cm[0, 0]
            writer.writerow([class_idx, tp, fp, fn, tn])

    # クラス数×クラス数の混同行列を保存
    class_cm_file = base_path / f'{methods}_overall_class_confusion_matrix.csv'
    with open(class_cm_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # ヘッダー行
        writer.writerow(['True/Pred'] + [f'Pred_{i}' for i in range(len(overall_metrics['class_confusion_matrix']))])
        # データ行
        for i, row in enumerate(overall_metrics['class_confusion_matrix']):
            writer.writerow([f'True_{i}'] + row.tolist())