from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import csv
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class BaseMultiLabelDataset(Dataset):
    """
    マルチラベルデータセットの基底クラス。
    共通のロジックを実装し、派生クラスで特定の動作を定義する。
    """
    def __init__(
        self,
        dataset_root: str,
        data_dirs: list[str],
        transform: transforms.Compose,
        num_classes: int,
    ) -> None:
        """
        初期化メソッド。

        Args:
            dataset_root (str): データセットのルートディレクトリ。
            transform (Callable): 画像に適用する変換関数。
            num_classes (int): クラス数。
        """
        self.dataset_root = Path(dataset_root)
        self.data_dirs = data_dirs
        self.transform = transform
        self.num_classes = num_classes
        self.image_dict: dict[str, list[int]] = {}  # 画像パスとラベルの辞書

        self._load_labels()

    def _load_labels(self) -> None:
        """ラベルを読み込む共通メソッド。"""
        for data_dir in self.data_dirs:
            self._load_labels_from_csv(self.dataset_root / f"{data_dir}.csv")

    def _load_labels_from_csv(self, csv_path: Path) -> None:
        """CSVファイルからラベルを読み込む。"""
        with open(csv_path, mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                labels = [int(label) for label in row[1:] if label.strip().isdigit()]
                labels = self._filter_labels(labels)
                labels.sort()  # ラベルを昇順にソート
                self.image_dict[self.dataset_root / csv_path.stem / row[0]] = labels

    def _filter_labels(self, labels: list[int]) -> list[int]:
        """
        ラベルをフィルタリングする。
        派生クラスで必要に応じてオーバーライドする。
        """
        if self.num_classes == 2:
            # 0~13のラベルを0に置き換え，14のラベルを1に置き換える
            return [0 if 0 <= label <= 13 else 1 for label in labels]
        return labels

    def __len__(self) -> int:
        """データセットのサイズを返す。"""
        return len(self.image_dict)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, torch.Tensor]:
        """
        指定されたインデックスのデータを取得する。

        Args:
            idx (int): データのインデックス。

        Returns:
            Tuple[Tensor, str, Tensor]: (画像テンソル, 画像パス, one-hotエンコードされたラベル)
        """
        image_path = list(self.image_dict.keys())[idx]
        labels = self.image_dict[image_path]

        # 画像を読み込む
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # ラベルをone-hotエンコード
        one_hot_label = torch.zeros(self.num_classes, dtype=torch.float)
        for label in labels:
            one_hot_label[label] = 1

        return image, str(image_path), one_hot_label