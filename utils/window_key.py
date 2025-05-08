from dataclasses import dataclass
from typing import Dict, List

@dataclass
class WindowSizeKey:
    """ウィンドウサイズの管理を行うクラス"""
    
    @staticmethod
    def create(size: int) -> str:
        """
        ウィンドウサイズからキーを生成
        
        Args:
            size (int): ウィンドウサイズ
            
        Returns:
            str: 生成されたキー
        """
        return f'sliding_window_{size}'

    @staticmethod
    def parse(key: str) -> int:
        """
        キーからウィンドウサイズを取得
        
        Args:
            key (str): ウィンドウサイズのキー
            
        Returns:
            int: ウィンドウサイズ
            
        Raises:
            ValueError: キーの形式が不正な場合
        """
        try:
            return int(key.split('_')[-1])
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid window size key format: {key}") from e

    @staticmethod
    def initialize_results(window_sizes: List[int]) -> Dict[str, dict]:
        """
        指定されたウィンドウサイズの結果保存用辞書を初期化
        
        Args:
            window_sizes (List[int]): ウィンドウサイズのリスト
            
        Returns:
            Dict[str, dict]: 初期化された結果保存用辞書
        """
        return {WindowSizeKey.create(size): {} for size in window_sizes}