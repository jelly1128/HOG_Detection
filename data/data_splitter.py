from dataclasses import dataclass
from enum import Enum

class DatasetType(Enum):
    """データセットの種類を表す列挙型"""
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

@dataclass
class SplitConfig:
    """分割設定を保持するデータクラス"""
    TRAIN_SIZE: int = 2  # 学習用に使用するsplitの数
    VAL_SIZE: int = 1    # 検証用に使用するsplitの数
    TEST_SIZE: int = 1   # テスト用に使用するsplitの数

class CrossValidationSplitter:
    def __init__(self, splits: dict):
        """
        
        """
        self.splits = splits
        self.n_splits = len(self.splits)
        self.split_config = SplitConfig()
        
        # 分割数の妥当性チェック
        total_splits = (self.split_config.TRAIN_SIZE + 
                       self.split_config.VAL_SIZE + 
                       self.split_config.TEST_SIZE)
        if total_splits != self.n_splits:
            raise ValueError(
                f"Total splits ({total_splits}) must equal number of available splits ({self.n_splits})"
            )

    def _get_split_name(self, index: int) -> str:
        """split名を生成する（1-basedインデックス）"""
        return f"split{index + 1}"
    
    def _get_circular_indices(self, start_idx: int, size: int) -> list[int]:
        """循環インデックスを取得する"""
        return [(start_idx + i) % self.n_splits for i in range(size)]

    def get_fold_splits(self) -> list[dict[str, list[str]]]:
        """各foldにおけるtrain/val/testの分割を取得"""
        split_names = [self._get_split_name(i) for i in range(self.n_splits)]
        fold_splits = []
        
        # 各splitをテストデータとして1回ずつ使用
        for test_idx in range(self.n_splits):
            # インデックスの取得
            train_indices = self._get_circular_indices(
                test_idx, 
                self.split_config.TRAIN_SIZE
            )
            val_indices = self._get_circular_indices(
                test_idx + self.split_config.TRAIN_SIZE,
                self.split_config.VAL_SIZE
            )
            
            # split名の取得
            train_splits = [split_names[i] for i in train_indices]
            val_split = split_names[val_indices[0]]
            test_split = split_names[(test_idx + self.n_splits - 1) % self.n_splits]
            
            fold_splits.append({
                DatasetType.TRAIN.value: train_splits,
                DatasetType.VAL.value: val_split,
                DatasetType.TEST.value: test_split
            })
                
        return fold_splits
    
    def get_split_folders(self) -> list[dict[str, list[str]]]:
        """各foldのtrain/val/test用フォルダ名リストを取得"""
        fold_splits = self.get_fold_splits()
        split_folders_list = []
        
        for fold in fold_splits:
            # 学習用splitに含まれるフォルダ名を結合
            train_folders = []
            for split_name in fold[DatasetType.TRAIN.value]:
                train_folders.extend(self.splits[split_name])
                
            # 検証用とテスト用のフォルダ名リストを取得
            val_folders = self.splits[fold[DatasetType.VAL.value]]
            test_folders = self.splits[fold[DatasetType.TEST.value]]
            
            split_folders_list.append({
                DatasetType.TRAIN.value: train_folders,
                DatasetType.VAL.value: val_folders,
                DatasetType.TEST.value: test_folders
            })
            
        return split_folders_list

def print_fold_summary(splitter: CrossValidationSplitter):
    """分割の要約を表示する補助関数"""
    split_folders = splitter.get_split_folders()
    fold_splits = splitter.get_fold_splits()
    
    for fold_idx, (folders, splits) in enumerate(zip(split_folders, fold_splits), 1):
        print(f"\nFold {fold_idx}:")
        print(f"Train splits: {splits[DatasetType.TRAIN.value]}")
        print(f"Val split: {splits[DatasetType.VAL.value]}")
        print(f"Test split: {splits[DatasetType.TEST.value]}")
        print(f"Number of train folders: {len(folders[DatasetType.TRAIN.value])}")
        print(f"Number of val folders: {len(folders[DatasetType.VAL.value])}")
        print(f"Number of test folders: {len(folders[DatasetType.TEST.value])}")

# 使用例
if __name__ == "__main__":
    splits_dict = {
            'split1': ["20210119093456_000001-001",
                        "20210531112330_000005-001",
                        "20211223090943_000001-002",
                        "20230718-102254-ES06_20230718-102749-es06-hd",
                        "20230802-104559-ES09_20230802-105630-es09-hd",
                        ],
            'split2': ["20210119093456_000001-002",
                        "20210629091641_000001-002",
                        "20211223090943_000001-003",
                        "20230801-125025-ES06_20230801-125615-es06-hd",
                        "20230803-110626-ES06_20230803-111315-es06-hd"
                        ],
            'split3': ["20210119093456_000002-001",
                        "20210630102301_000001-002",
                        "20220322102354_000001-002",
                        "20230802-095553-ES09_20230802-101030-es09-hd",
                        "20230803-093923-ES09_20230803-094927-es09-hd",
                        ],
            'split4': ["20210524100043_000001-001",
                        "20210531112330_000001-001",
                        "20211021093634_000001-001",
                        "20211021093634_000001-003"]
    }
    splitter = CrossValidationSplitter(splits_dict)
    print_fold_summary(splitter)