import csv
from engine.inference import InferenceResult
from labeling.label_converter import HardMultiLabelResult
from pathlib import Path
import numpy as np
from scipy.stats import entropy
import svgwrite
import re
from evaluate.metrics import ClassificationMetricsCalculator
from evaluate.save_metrics import save_video_metrics_to_csv, save_overall_metrics_to_csv
from sklearn.metrics import multilabel_confusion_matrix, classification_report


class ResultLoader:
    # 推論結果を読み込む
    def load_inference_results(self, csv_path: Path) -> InferenceResult:
        """
        CSVファイルから推論結果を読み込む。

        Args:
            csv_path: 読み込むCSVファイルのパス

        Returns:
            InferenceResult: 読み込んだ推論結果
        """
        try:
            image_paths, probabilities, labels = self._read_inference_results_csv(csv_path)
            video_result = InferenceResult(image_paths=image_paths, probabilities=probabilities, labels=labels)
            return video_result

        except Exception as e:
            raise

    def _read_inference_results_csv(self, csv_path: Path):
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
    

    def load_hard_multi_labels(self, csv_path: Path) -> HardMultiLabelResult:
        """
        CSVファイルからマルチラベルの結果を読み込む。
        Args:
            csv_path: 読み込むCSVファイルのパス
        Returns:
            HardMultiLabelResult: 読み込んだマルチラベルの結果
        """
        try:
            image_paths, multi_labels, ground_truth_labels = self._read_hard_multi_labels_results_csv(csv_path)
            hard_multi_label_result = HardMultiLabelResult(image_paths=image_paths, 
                                                           multi_labels=multi_labels, 
                                                           ground_truth_labels=ground_truth_labels)
            return hard_multi_label_result
        except Exception as e:
            raise

    
    def _read_hard_multi_labels_results_csv(self, csv_path: Path):
        """CSVファイルを読み込み、データを分割するヘルパーメソッド

        Args:
            csv_path (Path): 読み込むCSVファイルのパス

        Returns:
            tuple:
                list[str]: 画像パスのリスト
                list[list[int]]: マルチラベルのリスト
                list[list[int]]: ラベルのリスト
        """
        image_paths = []
        multi_labels = []
        ground_truth_labels = []

        with open(csv_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # ヘッダーをスキップ

            for row in reader:
                image_paths.append(row[0])
                # 確率値とラベルの区切り位置を計算
                num_multi_labels = (len(row) - 1) // 2
                
                # 確率値とラベルを分割して追加
                multi_labels.append(
                    list(map(float, row[1:num_multi_labels + 1]))
                )
                ground_truth_labels.append(
                    list(map(int, row[num_multi_labels + 1:]))
                )

        return image_paths, multi_labels, ground_truth_labels


class HOGDetectionVisualizer:
    """
    HOG検出器の結果を可視化するクラス。
    """
    def __init__(self, save_dir_path: Path, num_classes: int):
        self.save_dir_path = save_dir_path
        self.save_dir_path.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.label_color = {
            0: (254, 195, 195),  # white
            1: (204, 66, 38),    # lugol 
            2: (57, 103, 177),   # indigo
            3: (96, 165, 53),    # nbi
            4: (86, 65, 72),     # outside
            5: (159, 190, 183),  # bucket
            14: (0, 0, 0), # biopsy forceps
        }
        self.default_color = (255, 255, 255)

    def visualize(self,
                video_name: str,
                hard_multi_label_result: HardMultiLabelResult,
                with_ground_truth: bool = False,
                with_treatment: bool = False,
                ):
        """
        HOG検出器の結果を可視化するメソッド。
        """
        # 保存パスの設定
        video_results_dir = self.save_dir_path / video_name
        video_results_dir.mkdir(parents=True, exist_ok=True)

        # 予測結果の可視化
        self._create_svg_document(str(video_results_dir / f'{video_name}_prediction.svg'), hard_multi_label_result.multi_labels)

        # Ground Truthの可視化
        if with_ground_truth:
            self._create_svg_document(str(video_results_dir / f'{video_name}_ground_truth.svg'), hard_multi_label_result.ground_truth_labels)

        
    def _create_svg_document(self, document_name: str, multi_labels: list[list[int]]):
        # マルチラベルを取得
        multi_labels_np = np.array(multi_labels)
        n_images = len(multi_labels_np)
        
        # 時系列の画像を作成
        timeline_width = n_images
        timeline_height = n_images // 10
        
        # SVGドキュメントの作成
        dwg = svgwrite.Drawing(document_name, size=(timeline_width, timeline_height))
        dwg.add(dwg.rect((0, 0), (timeline_width, timeline_height), fill='white'))

        for i in range(n_images):
            image_labels = multi_labels_np[i]
            for label_idx, label_value in enumerate(image_labels):
                # if label_idx == 14 and label_value == 1:
                x1 = i * (timeline_width // n_images)
                x2 = (i + 1) * (timeline_width // n_images)
                y1 = 0
                y2 = n_images // 10
                if label_idx == 14 and label_value == 1:
                    color = self.label_color[label_idx]
                else:
                    color = self.default_color
                # RGBをSVG用の16進数カラーコードに変換
                color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                dwg.add(dwg.rect((x1, y1), (x2-x1, y2-y1), fill=color_hex))
        dwg.save()


class DatasetAnalyzer:
    """データセットの統計情報を分析・保存するクラス"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_class_distribution(self, video_results: dict):
        """
        動画ごとのクラス分布を分析し、CSVファイルに保存する
        
        Args:
            video_results (dict): {video_name: HardMultiLabelResult} 形式の辞書
        """
        import pandas as pd
        from collections import Counter
        import json
        
        # 全動画の統計情報を格納する辞書
        stats = {}
        
        for video_name, result in video_results.items():
            # ground truthラベルを取得
            labels = result.ground_truth_labels
            
            # クラスごとのカウンターを初期化
            class_counter = Counter()
            
            # 各フレームのラベルをカウント
            for frame_labels in labels:
                for cls_idx, value in enumerate(frame_labels):
                    if value == 1:
                        class_counter[cls_idx] += 1
            
            total_frames = len(labels)
            
            # 動画の統計情報を保存
            stats[video_name] = {
                "total_frames": total_frames,
                "class_counts": dict(class_counter),
                "class_percentages": {
                    cls: (count / total_frames) * 100 
                    for cls, count in class_counter.items()
                }
            }
        
        # 統計情報をDataFrameに変換
        df_rows = []
        for video_name, video_stats in stats.items():
            row = {"video_name": video_name, "total_frames": video_stats["total_frames"]}
            
            # クラスごとのカウントとパーセンテージを追加
            for cls in range(15):  # 15クラスを想定
                count = video_stats["class_counts"].get(cls, 0)
                percentage = video_stats["class_percentages"].get(cls, 0)
                row[f"class_{cls}_count"] = count
                row[f"class_{cls}_percentage"] = f"{percentage:.2f}%"
            
            df_rows.append(row)
        
        # DataFrameを作成してCSVに保存
        df = pd.DataFrame(df_rows)
        csv_path = self.save_dir / "class_distribution.csv"
        df.to_csv(csv_path, index=False)
        print(f"クラス分布の統計情報を保存しました: {csv_path}")
        
        # 詳細な統計情報をJSONとしても保存
        json_path = self.save_dir / "class_distribution_detailed.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        print(f"詳細な統計情報を保存しました: {json_path}")
        
        return stats


def show_dataset_stats(hard_multi_label_result: HardMultiLabelResult, save_dir: Path):
    """
    動画ごとのクラスごとのサンプル数をカウントして表示・保存する関数。

    Args:
        hard_multi_label_result (HardMultiLabelResult): HardMultiLabelResultオブジェクト。
        save_dir (Path): 統計情報を保存するディレクトリ。
    """
    from collections import Counter
    import pandas as pd

    # 保存ディレクトリの作成
    save_dir.mkdir(parents=True, exist_ok=True)

    # 統計情報を格納するリスト
    stats = []

    # ground_truth_labelsを取得
    ground_truth_labels = hard_multi_label_result.ground_truth_labels

    # クラスごとのサンプル数をカウント
    class_counter = Counter()
    for frame_labels in ground_truth_labels:
        for cls_idx, value in enumerate(frame_labels):
            if value == 1:
                class_counter[cls_idx] += 1

    # 統計情報をリストに追加
    stats.append({
        **{f"class_{cls}_count": class_counter.get(cls, 0) for cls in range(15)}  # 15クラスを想定
    })

    # DataFrameに変換
    df = pd.DataFrame(stats)

    # CSVに保存
    csv_path = save_dir / "dataset_class_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"データセットのクラス統計情報を保存しました: {csv_path}")

    # 統計情報を表示
    print(df)


def analyze_class14_fp_fn_confusion(hard_multi_label_result, save_dir=None, video_name=None):
    """
    クラス14のFP（正解0,予測1）・FN（正解1,予測0）で、どのクラスと間違えたかを集計する。
    Args:
        hard_multi_label_result (HardMultiLabelResult): 対象データ
        save_dir (Path or None): 結果をCSV保存する場合のディレクトリ
        video_name (str or None): ファイル名用
    Returns:
        dict: {'FP': {クラスidx: カウント}, 'FN': {クラスidx: カウント}}
    """
    import pandas as pd
    from collections import Counter
    
    multi_labels = hard_multi_label_result.multi_labels
    ground_truth_labels = hard_multi_label_result.ground_truth_labels
    n_classes = len(multi_labels[0])
    
    fp_counter = Counter()
    fn_counter = Counter()
    
    for pred, true in zip(multi_labels, ground_truth_labels):
        # FP: 正解0, 予測1
        if true[14] == 0 and pred[14] == 1:
            # 他クラスで予測1のものをカウント
            for i in range(n_classes):
                if i != 14 and pred[i] == 1:
                    fp_counter[i] += 1
        # FN: 正解1, 予測0
        if true[14] == 1 and pred[14] == 0:
            # 他クラスで予測1のものをカウント
            for i in range(n_classes):
                if i != 14 and pred[i] == 1:
                    fn_counter[i] += 1
    
    result = {'FP': dict(fp_counter), 'FN': dict(fn_counter)}
    
    # CSV保存
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {**{'type': 'FP'}, **result['FP']},
        {**{'type': 'FN'}, **result['FN']}
    ])
    fname = f'class14_fp_fn_confusion.csv' if video_name is None else f'{video_name}_class14_fp_fn_confusion.csv'
    df.to_csv(save_dir / fname, index=False)
    
    # 結果をprint
    print(f"\n=== クラス14のFPで他クラスと間違えた数 ===\n{dict(fp_counter)}")
    print(f"\n=== クラス14のFNで他クラスと間違えた数 ===\n{dict(fn_counter)}")
    return result


def main():
    save_dir_path = Path("15class_resnet50_r_s_s_multitask_test_hog")
    save_dir_path.mkdir(parents=True, exist_ok=True)
    # 解析器の初期化
    
    calculator = ClassificationMetricsCalculator(num_classes=15)

    result_loader = ResultLoader()
    
    # 全体のディレクトリパス
    dir_path = Path("15class_resnet50_r_s_s_multitask_test")

    # 全体の結果を保存する辞書
    overall_results = {}

    # フォルダを走査
    for i in range(4):  # fold_0からfold_3まで
        # 数字から始まる名前のフォルダを取得
        video_dirs = dir_path.glob(f"fold_{i}/*")

        fold_results = {}
        
        for video_dir in video_dirs:
            # フォルダ名からフォルダ名を取得
            video_name = video_dir.name

            if re.match(r"^\d+", video_name):
                method = 'threshold_50%'
                raw_result_csv_path = dir_path / f'fold_{i}' / video_name / method / f'{method}_results_{video_name}.csv'

                # 推論結果を読み込む
                hard_multi_label_result = result_loader.load_hard_multi_labels(raw_result_csv_path)
                # 可視化
                save_fold_dir_path = save_dir_path / f'fold_{i}'
                save_fold_dir_path.mkdir(parents=True, exist_ok=True)
                visualizer = HOGDetectionVisualizer(save_dir_path=save_fold_dir_path, num_classes=15)
                visualizer.visualize(video_name, hard_multi_label_result, with_ground_truth=True)
                # # 統計情報を表示・保存
                # show_dataset_stats(hard_multi_label_result, save_dir_path / video_name)

                # # クラス14の誤分類解析
                # analyze_class14_fp_fn_confusion(hard_multi_label_result, save_dir=save_dir_path / video_name, video_name=video_name)

                fold_results[video_name] = hard_multi_label_result
                overall_results[video_name] = hard_multi_label_result

        video_metrics = calculator.calculate_multi_label_metrics_per_video(fold_results)
        overall_metrics = calculator.calculate_multi_label_overall_metrics(fold_results)
        ## 各動画フォルダにマルチラベルのメトリクスを保存
        save_video_metrics_to_csv(video_metrics, save_dir_path / f'fold_{i}', methods = 'hog_detection')
        ## 全体のメトリクスを保存
        save_overall_metrics_to_csv(overall_metrics, save_dir_path / f'fold_{i}', methods = 'hog_detection')


    # 全体のメトリクスを計算
    overall_video_metrics = calculator.calculate_multi_label_metrics_per_video(overall_results)
    overall_metrics = calculator.calculate_multi_label_overall_metrics(overall_results)
    ## 各動画フォルダにマルチラベルのメトリクスを保存 
    save_video_metrics_to_csv(overall_video_metrics, save_dir_path, methods = 'hog_detection')
    save_overall_metrics_to_csv(overall_metrics, save_dir_path / 'overall_results', methods = 'hog_detection')

if __name__ == "__main__":
    main()