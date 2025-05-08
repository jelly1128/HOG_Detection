from pathlib import Path
import torch
from torch.utils.data import DataLoader
import csv
from dataclasses import dataclass

# 推論結果を格納するクラス
@dataclass
class InferenceResult:
    """
    推論結果を格納するクラス。

    Attributes:
        image_paths: 画像パスのリスト
        probabilities: 予測確率のリスト
        labels: 正解ラベルのリスト
    """
    image_paths: list[str]
    probabilities: list[list[float]]
    labels: list[list[int]]

class Inference:
    def __init__(self, model: torch.nn.Module, device: str):
        """
        推論エンジンの初期化。

        Args:
            model: 推論に使用するモデル
            device: 使用するデバイス（'cuda' または 'cpu'）
        """
        self.model = model
        self.device = device
        self.model.eval()  # モデルを評価モードに設定

    def _run_inference(self, test_dataloader: DataLoader) -> InferenceResult:
        """
        1つのデータローダーに対して推論を実行する。

        Args:
            test_dataloader: テスト用データローダー

        Returns:
            InferenceResult: 推論結果
            - image_paths: 画像パスのリスト
            - probabilities: 予測確率のリスト
            - labels: 正解ラベルのリスト
        """
        results = InferenceResult(image_paths=[], probabilities=[], labels=[])
        
        with torch.no_grad():
            for images, image_paths, labels in test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                sigmoid_outputs = torch.sigmoid(outputs[0])
                
                probabilities = [round(float(prob), 4) for prob in sigmoid_outputs.tolist()]
                labels = labels.cpu().numpy().astype(int).tolist()
                
                results.image_paths.extend(image_paths)
                results.probabilities.append(probabilities)
                results.labels.extend(labels)
        
        return results

    def _save_results(self, save_dir_path: Path, video_name: str, results: InferenceResult):
        """
        推論結果を保存する。

        Args:
            save_dir_path: 保存先のディレクトリパス
            video_name: テスト動画のフォルダ名
            results: (probabilities, labels, image_paths)のタプル
        """
        save_path = save_dir_path / video_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 結果をCSVファイルに保存
        self._save_raw_results(save_path / f'raw_results_{video_name}.csv', results)
        
    def _save_raw_results(self, csv_path: Path, results: InferenceResult):
        """
        推論結果をCSVファイルに保存する。
        
        Args:
            csv_path: 保存先のパス
            results: 推論結果
                image_paths: 画像パスのリスト
                probabilities: 予測確率のリスト
                labels: 正解ラベルのリスト
        """
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Image_Path'] + [f"Pred_Class_{i}" for i in range(len(results.probabilities[0]))] + \
                    [f"True_Class_{i}" for i in range(len(results.labels[0]))]
            writer.writerow(header)
            for img_path, probs, label in zip(results.image_paths, results.probabilities, results.labels):
                writer.writerow([img_path] + probs + label)
        
    def run(self, save_dir_path: str, test_dataloaders: dict[str, DataLoader]) -> dict[str, InferenceResult]:
        """
        推論を実行し、モデルの出力を返す。

        Args:
            save_dir: 結果保存ディレクトリ
            test_dataloaders: テスト用データローダーの辞書

        Returns:
            results: フォルダごとの推論結果を格納した辞書
            - video_name: フォルダ名
            - results: 推論結果
        """
        results = {}
        
        for video_name, test_dataloader in test_dataloaders.items():
            
            # 推論実行
            folder_results = self._run_inference(test_dataloader)
            
            # 結果を保存
            self._save_results(save_dir_path, video_name, folder_results)
            
            # 結果を辞書に格納
            results[video_name] = folder_results

        return results