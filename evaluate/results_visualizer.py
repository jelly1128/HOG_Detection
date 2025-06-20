import numpy as np
from pathlib import Path
import csv
import svgwrite
from engine.inference import InferenceResult
from labeling.result_types import HardMultiLabelResult, SingleLabelResult

# 定数の定義
LABEL_COLORS = {
    0: (255, 255, 255),  # 
    1: (0, 0, 0),        # biopsy forceps
}
DEFAULT_COLOR = (255, 255, 255)

class ResultsVisualizer:
    def __init__(self, save_dir_path: Path):
        """
        結果の可視化を行うクラス

        Args:
            save_dir: 可視化結果の保存ディレクトリ
        """
        self.save_dir_path = save_dir_path
        self.save_dir_path.mkdir(parents=True, exist_ok=True)
        
    def load_results(self, csv_path: Path) -> InferenceResult:
        """
        CSVファイルから推論結果を読み込む。

        Args:
            csv_path: 読み込むCSVファイルのパス

        Returns:
            InferenceResult: 読み込んだ推論結果
        """
        try:
            image_paths, probabilities, labels = self._read_csv(csv_path)
            return InferenceResult(image_paths=image_paths, probabilities=probabilities, labels=labels)
        except Exception as e:
            raise
    
    def _read_csv(self, csv_path: Path):
        """CSVファイルを読み込み、データを分割するヘルパーメソッド
        
        Args:
            csv_path (Path): 読み込むCSVファイルのパス
            
        Returns:
            tuple:
                list[str]: 画像パスのリスト
                list[list[float]]: 確率値のリスト
                list[list[int]]: ラベルのリスト
        """
        image_paths = []
        probabilities = []
        ground_truth_labels = []

        with open(csv_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # ヘッダーをスキップ

            for row in reader:
                image_paths.append(row[0])
                # 確率値とラベルの区切り位置を計算
                num_probabilities = (len(row) - 1) // 2
                
                # 確率値とラベルを分割して追加
                probabilities.append(
                    list(map(float, row[1:num_probabilities + 1]))
                )
                ground_truth_labels.append(
                    list(map(int, row[num_probabilities + 1:]))
                )

        return image_paths, probabilities, ground_truth_labels
      
    def save_multi_label_visualization(self, 
                                       results: dict[str, HardMultiLabelResult], 
                                       save_path: Path = None, 
                                       methods: str = 'multi_label'
                                       ):
        """
        マルチラベル分類の予測結果を時系列で可視化
        
        Args:
            results (dict[str, HardMultiLabelResult]): マルチラベルの予測結果
            save_path (Path, optional): 保存先のパス。指定しない場合はself.save_dirを使用。
            methods (str, optional): メソッド名。デフォルトは'multi_label'。
        """
        for video_name, result in results.items():
            # マルチラベルの予測結果を取得
            predicted_labels = np.array(result.multi_labels)
            n_images = len(predicted_labels)
            n_classes = len(predicted_labels[0])
            
            # 時系列の画像を作成
            timeline_width = n_images
            timeline_height = n_classes * (n_images // 10)

            # 保存パスの設定
            if save_path is None:
                video_results_dir = self.save_dir_path / video_name / methods
                video_results_dir.mkdir(parents=True, exist_ok=True)
                save_file = video_results_dir / f'{methods}_{video_name}.svg'
            else:
                save_path = save_path / video_name / methods
                save_path.mkdir(parents=True, exist_ok=True)
                save_file = save_path / f'{methods}_{video_name}.svg'
            
            # SVGドキュメントの作成
            dwg = svgwrite.Drawing(str(save_file), size=(timeline_width, timeline_height))
            dwg.add(dwg.rect((0, 0), (timeline_width, timeline_height), fill='white'))

            for i in range(n_images):
                labels = predicted_labels[i]
                for label_idx, label_value in enumerate(labels):
                    if label_value == 1:
                        x1 = i * (timeline_width // n_images)
                        x2 = (i + 1) * (timeline_width // n_images)
                        y1 = label_idx * (n_images // 10)
                        y2 = (label_idx + 1) * (n_images // 10)

                        color = LABEL_COLORS.get(label_idx, DEFAULT_COLOR)
                        # RGBをSVG用の16進数カラーコードに変換
                        color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                        dwg.add(dwg.rect((x1, y1), (x2-x1, y2-y1), fill=color_hex))

            # SVGファイルを保存
            dwg.save()
            
    def save_single_label_visualization(self, results: dict[str, SingleLabelResult], save_path: Path = None, methods: str = 'single_label'):
        """シングルラベル分類の予測結果を時系列で可視化"""
        for video_name, result in results.items():
            # シングルラベルの予測結果を取得
            predicted_labels = np.array(result.single_labels)
            n_images = len(predicted_labels)
            
            # 時系列の画像を作成
            timeline_width = n_images
            timeline_height = n_images // 10
            
            # 保存パスの設定
            if save_path is None:
                video_results_dir = self.save_dir_path / video_name / methods
                video_results_dir.mkdir(parents=True, exist_ok=True)
                save_file = video_results_dir / f'{methods}_{video_name}.svg'
            else:
                save_path = save_path / video_name
                save_path.mkdir(parents=True, exist_ok=True)
                save_file = save_path / f'{methods}_{video_name}.svg'
                
            # SVGドキュメントの作成
            dwg = svgwrite.Drawing(str(save_file), size=(timeline_width, timeline_height))
            dwg.add(dwg.rect((0, 0), (timeline_width, timeline_height), fill='white'))

            for i in range(n_images):
                label = predicted_labels[i]
                x1 = i * (timeline_width // n_images)
                x2 = (i + 1) * (timeline_width // n_images)
                
                color = LABEL_COLORS.get(label, DEFAULT_COLOR)
                color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                dwg.add(dwg.rect((x1, 0), (x2-x1, timeline_height), fill=color_hex))

            # SVGファイルを保存
            dwg.save()
            
    def save_main_classes_visualization(self, results: dict[str, HardMultiLabelResult], save_path: Path = None):
        """正解ラベルを時系列で可視化"""
        for video_name, result in results.items():
            # 主クラス（0-5）のみを抽出
            ground_truth_labels = np.array(result.ground_truth_labels)[:, :6]  # 最初の6クラスのみを抽出
            n_images = len(ground_truth_labels)
            
            # 時系列の画像を作成
            timeline_width = n_images
            timeline_height = n_images // 10
            
            # 保存パスの設定
            if save_path is None:
                video_results_dir = self.save_dir_path / video_name
                video_results_dir.mkdir(parents=True, exist_ok=True)
                save_file = video_results_dir / f'main_classes_{video_name}.svg'
            else:
                save_path = save_path / video_name
                save_path.mkdir(parents=True, exist_ok=True)
                save_file = save_path / f'main_classes_{video_name}.svg'

            # SVGドキュメントの作成
            dwg = svgwrite.Drawing(str(save_file), size=(timeline_width, timeline_height))
            dwg.add(dwg.rect((0, 0), (timeline_width, timeline_height), fill='white'))

            for i in range(n_images):
                labels = ground_truth_labels[i]
                for label_idx, label_value in enumerate(labels):
                    if label_value == 1:
                        x1 = i * (timeline_width // n_images)
                        x2 = (i + 1) * (timeline_width // n_images)

                        color = LABEL_COLORS.get(label_idx, DEFAULT_COLOR)
                        color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                        dwg.add(dwg.rect((x1, 0), (x2-x1, timeline_height), fill=color_hex))

            # SVGファイルを保存
            dwg.save()

    def save_single_label_ground_truth_visualization(self, results: dict[str, SingleLabelResult], save_path: Path = None, methods: str = 'single_label_gt'):
        """
        シングルラベル分類の正解ラベル（ground truth）を時系列で可視化
        Args:
            results (dict[str, SingleLabelResult]): シングルラベルの推論結果（ground_truth_labelsを利用）
            save_path (Path, optional): 保存先のパス。指定しない場合はself.save_dirを使用。
            methods (str, optional): メソッド名。デフォルトは'single_label_gt'。
        """
        for video_name, result in results.items():
            ground_truth_labels = np.array(result.ground_truth_labels)
            n_images = len(ground_truth_labels)

            timeline_width = n_images
            timeline_height = n_images // 10

            # 保存パスの設定
            if save_path is None:
                video_results_dir = self.save_dir_path / video_name / methods
                video_results_dir.mkdir(parents=True, exist_ok=True)
                save_file = video_results_dir / f'{methods}_{video_name}.svg'
            else:
                save_path = save_path / video_name
                save_path.mkdir(parents=True, exist_ok=True)
                save_file = save_path / f'{methods}_{video_name}.svg'

            # SVGドキュメントの作成
            import svgwrite
            dwg = svgwrite.Drawing(str(save_file), size=(timeline_width, timeline_height))
            dwg.add(dwg.rect((0, 0), (timeline_width, timeline_height), fill='white'))

            for i in range(n_images):
                label = ground_truth_labels[i]
                x1 = i * (timeline_width // n_images)
                x2 = (i + 1) * (timeline_width // n_images)
                color = LABEL_COLORS.get(label, DEFAULT_COLOR)
                color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                dwg.add(dwg.rect((x1, 0), (x2-x1, timeline_height), fill=color_hex))

            dwg.save()