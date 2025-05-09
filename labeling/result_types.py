from dataclasses import dataclass

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

@dataclass
class SingleLabelResult:
    """
    シングルラベルの結果を格納するクラス。
    Attributes:
        image_paths: 画像パスのリスト
        single_labels: シングルラベルの予測ラベルのリスト
        ground_truth_labels: シングルラベルの正解ラベルのリスト
    """
    image_paths: list[str]
    single_labels: list[int]
    ground_truth_labels: list[int]

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
