import logging
from pathlib import Path
import time

# 日本の時刻に設定
def jst_time(*args):
    return time.localtime(time.time() + 9 * 3600)  # UTC+9

logging.Formatter.converter = jst_time


def setup_logging(log_dir: Path, mode: str) -> logging.Logger:
    """
    ロギングの設定を行う関数
    
    Args:
        log_dir (Path): ログファイルを保存するディレクトリ
        mode (str): ロギングモード（例：'training_fold_0'）
    
    Returns:
        logging.Logger: 設定済みのロガーオブジェクト
    """
    # ロガーの取得
    logger = logging.getLogger(mode)  # モードごとに異なるロガーを作成
    logger.setLevel(logging.INFO)
    
    # 既存のハンドラをクリア
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # フォーマッタの作成
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # ファイルハンドラの設定
    log_file = log_dir / f"{mode}.log"
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # コンソールハンドラの設定
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Logs will be saved in {log_file}")
    
    return logger