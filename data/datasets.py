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
        if self.num_classes == 6:
            # 0～5のクラスのみを使用
            return [label for label in labels if 0 <= label <= 5]
        elif self.num_classes == 7:
            # 6～14のラベルを6に置き換える
            return [6 if 6 <= label <= 14 else label for label in labels]
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

class BaseSingleLabelDataset(Dataset):
    """
    シングルラベルデータセットの基底クラス。
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
            data_dirs (list[str]): 使用するサブディレクトリ名リスト。
            transform (Callable): 画像に適用する変換関数。
            num_classes (int): クラス数。
        """
        self.dataset_root = Path(dataset_root)
        self.data_dirs = data_dirs
        self.transform = transform
        self.num_classes = num_classes
        self.image_dict: dict[str, int] = {}  # 画像パスとラベル（整数）の辞書

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
                label = self._filter_label(row)
                self.image_dict[str(self.dataset_root / csv_path.stem / row[0])] = label

    def _filter_label(self, row: list[str]) -> int:
        """
        ラベルをフィルタリングする。
        派生クラスで必要に応じてオーバーライドする。
        デフォルトは1つ目のラベルを整数として返す。
        """
        return int(row[1])

    def __len__(self) -> int:
        """データセットのサイズを返す。"""
        return len(self.image_dict)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        """
        指定されたインデックスのデータを取得する。

        Args:
            idx (int): データのインデックス。

        Returns:
            Tuple[Tensor, str, int]: (画像テンソル, 画像パス, 整数ラベル)
        """
        image_path = list(self.image_dict.keys())[idx]
        label = self.image_dict[image_path]

        # 画像を読み込む
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # シングルラベルの場合は整数ラベルを返す（one-hotにはしない）
        return image, str(image_path), label

class CustomSingleLabelDataset(BaseSingleLabelDataset):
    """
    マルチラベルcsvをシングルラベル化して読み込むカスタムデータセット。
    指定したラベル群を1つのラベルに統一し、それ以外は0~3のラベルをそのまま使用。
    CSVのラベル列が1つだけなら整数ラベルとして扱い、複数ならmulti-hotとして扱う。

    Args:
        dataset_root (str): データセットのルートディレクトリ。
        data_dirs (list[str]): 使用するサブディレクトリ名リスト。
        transform (Callable): 画像に適用する変換関数。
        num_classes (int): クラス数。
        merge_label_indices (list[int]): 統一するラベルのインデックスリスト。
        merge_to_label (int): 統一先のラベル。
    """
    def __init__(
        self,
        dataset_root: str,
        data_dirs: list[str],
        transform: transforms.Compose,
        num_classes: int,
        merge_label_indices: list[int] = [4, 5, 6, 11, 12],
        merge_to_label: int = 4,
    ) -> None:
        self.merge_label_indices = merge_label_indices
        self.merge_to_label = merge_to_label
        super().__init__(dataset_root, data_dirs, transform, num_classes)

    def _filter_label(self, row: list[str]) -> int:
        # ラベル1,ラベル2を取得（空欄は除外）
        labels = []
        for v in row[1:3]:
            v = v.strip()
            if v != '':
                try:
                    labels.append(int(v))
                except Exception:
                    continue
        # 4,5,6,11,12があれば4に統一
        if any(l in self.merge_label_indices for l in labels):
            return self.merge_to_label
        # 0～3があればその値を返す（優先順位:ラベル1→ラベル2）
        for l in labels:
            if l in [0,1,2,3]:
                return l
        raise ValueError(f"不正なラベル: {row}")

class HOGDetectionBinaryDataset(BaseSingleLabelDataset):
    """
    0～13を0、14を1に変換して読み込むHOG Detection用バイナリデータセット。

    Args:
        dataset_root (str): データセットのルートディレクトリ。
        data_dirs (list[str]): 使用するサブディレクトリ名リスト。
        transform (Callable): 画像に適用する変換関数。
    """
    def __init__(
        self,
        dataset_root: str,
        data_dirs: list[str],
        transform: transforms.Compose,
    ) -> None:
        # バイナリ分類なのでnum_classes=2
        super().__init__(dataset_root, data_dirs, transform, num_classes=2)

    def _filter_label(self, row: list[str]) -> int:
        """
        ラベル列から0～13なら0、14なら1に変換して返す。
        """
        # 複数ラベルがある場合も考慮
        labels = [int(v) for v in row[1:] if v.strip().isdigit()]
        if 14 in labels:
            return 1
        else:
            return 0