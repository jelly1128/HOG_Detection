import csv
import numpy as np
from engine.inference import InferenceResult
from labeling.label_converter import HardMultiLabelResult, SingleLabelResult
from pathlib import Path
from evaluate.metrics import ClassificationMetricsCalculator
from evaluate.results_visualizer import ResultsVisualizer
from utils.window_key import WindowSizeKey

class Analyzer:
    def __init__(self, save_dir_path: Path, num_classes: int):
        """
        推論結果を解析するクラス。

        Args:
            save_dir_path (str): 結果を保存するディレクトリ
            num_classes (int): クラス数
        """
        self.save_dir_path = save_dir_path
        self.num_classes = num_classes

    def apply_sliding_window_to_hard_multi_label_results(self, hard_multi_label_results: dict[str, HardMultiLabelResult], window_size=5, step=1):
        """
        マルチラベルの予測結果にスライディングウィンドウを適用して、主クラス（0-5）のシングルラベル予測に変換する関数

        Args:
            hard_multi_label_results (dict[str, HardMultiLabelResult]): 各フォルダのマルチラベルの結果
            window_size (int): スライディングウィンドウのサイズ
            step (int): スライディングウィンドウのステップ幅

        Returns:
            dict[str, SingleLabelResult]: 各フォルダの主クラスのシングルラベル予測結果
        """
        smoothed_results = {}

        for folder_name, result in hard_multi_label_results.items():
            y_pred = np.array(result.multi_labels)[:, :6]  # 主クラスのみ
            y_true = np.array(result.ground_truth_labels)[:, :6]  # 主クラスのみ
            num_frames = len(y_pred)

            # 予測ラベルにのみスライディングウィンドウを適用
            smoothed_labels = []
            for start in range(0, num_frames - window_size + 1, step):
                window_pred = y_pred[start:start + window_size]
                class_counts_pred = window_pred.sum(axis=0)
                smoothed_label = np.argmax(class_counts_pred)
                smoothed_labels.append(smoothed_label)

            # 正解ラベルは主クラスの中で1が立っているインデックスを取得
            true_labels = [np.argmax(label) for label in y_true[:len(smoothed_labels)]]

            smoothed_results[folder_name] = SingleLabelResult(
                image_paths=result.image_paths[:len(smoothed_labels)],
                single_labels=smoothed_labels,
                ground_truth_labels=true_labels
            )

        return smoothed_results

    def analyze_sliding_windows(self,
                                hard_multi_label_results: dict[str, HardMultiLabelResult], 
                                visualizer: ResultsVisualizer, 
                                calculator: ClassificationMetricsCalculator, 
                                window_sizes: list=None):
        """
        異なるウィンドウサイズでスライディングウィンドウ解析を実行し、結果をまとめる

        Args:
            hard_multi_label_results: マルチラベルの予測結果
            visualizer: 可視化用のインスタンス
            calculator: メトリクス計算用のインスタンス
            window_sizes: ウィンドウサイズのリスト（Noneの場合はデフォルト値を使用）

        Returns:
            dict: 各ウィンドウサイズの結果
            dict: 各ウィンドウサイズのメトリクス
        """
        if window_sizes is None:
            window_sizes = range(3, 16, 2)  # 3, 5, 7, 9, 11, 13, 15
        
        # 結果保存用の辞書を初期化
        all_window_results = WindowSizeKey.initialize_results(window_sizes)
        
        for window_size in window_sizes:
            window_key = WindowSizeKey.create(window_size)
            # スライディングウィンドウを適用
            sliding_window_results = self.apply_sliding_window_to_hard_multi_label_results(
                hard_multi_label_results, 
                window_size=window_size
            )

            # 結果を保存
            for folder_name, result in sliding_window_results.items():
                save_path = self.save_dir_path / folder_name / window_key
                save_path.mkdir(parents=True, exist_ok=True)

                with open(save_path / f'{window_key}_results_{folder_name}.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    header = ['Image_Path'] + [f"Pred_Class"] + [f"True_Class"]
                    writer.writerow(header)
                    # 1行ずつ書き込み
                    for image_path, pred_label, true_label in zip(
                        result.image_paths, 
                        result.single_labels, 
                        result.ground_truth_labels
                    ):
                        writer.writerow([image_path, pred_label, true_label])
            
            all_window_results[window_key] = sliding_window_results
        
        # サマリーファイルを作成
        # self.save_sliding_window_summary(all_window_metrics)
        
        return all_window_results