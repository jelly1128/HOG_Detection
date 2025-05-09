from dataclasses import dataclass
from labeling.result_types import InferenceResult
import csv
from pathlib import Path

@dataclass
class HardMultiLabelResult:
    """
    マルチラベルの結果を格納するクラス。

    Attributes:
        image_paths: 画像パスのリスト
        multi_labels: マルチラベルの予測ラベルのリスト
        ground_truth_labels: マルチラベルの正解ラベルのリスト
    """
    image_paths: list[str]
    multi_labels: list[list[int]]
    ground_truth_labels: list[list[int]]
    

class MultiToSingleLabelConverter:
    def __init__(self, inference_results_dict: dict[str, InferenceResult]):
        self.inference_results_dict = inference_results_dict
        
    def convert_soft_to_hard_multi_labels(self, threshold: float = 0.5) -> dict[str, HardMultiLabelResult]:
        """
        確率のしきい値を使用して、マルチソフトラベルをマルチラベルに変換します。
        全てのラベルが閾値を超えていない場合、主ラベルの中で最も確率が高いものとその確率より高いサブラベルを選択します。
        
        Args:
            threshold (float): ラベルの割り当てを決定する確率のしきい値。
        
        Returns:
            dict[str, HardMultiLabelResult]: しきい値を超えるラベルのリストを含む辞書。
        """
        hard_multi_labels_results = {}
        # フォルダごとにマルチソフトラベルをマルチラベルに変換
        for video_name, inference_result in self.inference_results_dict.items():
            hard_multi_label_result = HardMultiLabelResult(image_paths=[], multi_labels=[], ground_truth_labels=[])
            
            for image_path, probabilities, labels in zip(inference_result.image_paths, inference_result.probabilities, inference_result.labels):
                # 通常の閾値処理
                hard_multi_label = [1 if prob > threshold else 0 for prob in probabilities]
                
                # 全てのラベルが0の場合の処理
                if sum(hard_multi_label) == 0:
                    # 主ラベル（0-5）の中で最も確率が高いものを見つける
                    main_probs = probabilities[:6]  # 主ラベルの確率
                    max_main_prob = max(main_probs)
                    max_main_idx = main_probs.index(max_main_prob)
                    
                    # 主ラベルの最大確率より高い確率を持つサブラベルを見つける
                    hard_multi_label = [0] * len(probabilities)
                    hard_multi_label[max_main_idx] = 1  # 最も確率の高い主ラベルを1に設定
                    
                    # サブラベル（6以降）で主ラベルの最大確率より高いものを1に設定
                    for i, prob in enumerate(probabilities[6:], start=6):
                        if prob > max_main_prob:
                            hard_multi_label[i] = 1
                
                hard_multi_label_result.image_paths.append(image_path)
                hard_multi_label_result.multi_labels.append(hard_multi_label)
                hard_multi_label_result.ground_truth_labels.append(labels)
            
            hard_multi_labels_results[video_name] = hard_multi_label_result
            
        return hard_multi_labels_results
    
    def save_hard_multi_label_results(self, 
                                      hard_multi_label_results: dict[str, HardMultiLabelResult], 
                                      save_dir_path: Path, 
                                      methods: str = "threshold"):
        """
        マルチラベルの変換結果をCSVファイルに保存します。

        Args:
            hard_multi_label_results (dict[str, HardMultiLabelResult]): 変換結果の辞書
            save_dir (Path): 保存先ディレクトリ
            method (str): 変換手法の名前（ファイル名に使用）
        """
        # 保存先ディレクトリが存在しない場合は作成
        save_dir_path.mkdir(parents=True, exist_ok=True)
        
        # 動画ごとに結果を保存
        for video_name, result in hard_multi_label_results.items():
            video_results_dir = save_dir_path / video_name / methods
            video_results_dir.mkdir(parents=True, exist_ok=True)
            # フォルダごとのCSVファイルを作成
            video_results_csv = video_results_dir / f'{methods}_results_{video_name}.csv'
            
            with open(video_results_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                # ヘッダー行
                header = ['Image_Path']
                header.extend([f'Label_{i}' for i in range(len(result.multi_labels[0]))])  # 予測ラベル
                header.extend([f'True_Label_{i}' for i in range(len(result.ground_truth_labels[0]))])  # 正解ラベル
                writer.writerow(header)
                
                # データ行
                for img_path,           pred_labels,          true_labels in zip(
                    result.image_paths, result.multi_labels,  result.ground_truth_labels
                ):
                    row = [img_path]
                    row.extend(pred_labels)
                    row.extend(true_labels)
                    writer.writerow(row)