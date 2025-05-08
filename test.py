import argparse
import logging
from pathlib import Path


from config.config_loader import load_test_config
from data.data_splitter import CrossValidationSplitter
from data.dataloader import DataLoaderFactory
from data.dataset_visualizer import plot_dataset_samples, show_dataset_stats
from engine.inference import Inference
from evaluate.analyzer import Analyzer
from utils.torch_utils import get_device_and_num_gpus, set_seed
from utils.logger import setup_logging
from utils.window_key import WindowSizeKey
from evaluate.results_visualizer import ResultsVisualizer
from labeling.label_converter import MultiToSingleLabelConverter
from evaluate.metrics import ClassificationMetricsCalculator
from evaluate.save_metrics import save_video_metrics_to_csv, save_overall_metrics_to_csv
from model.setup_models import setup_model


def test(config: dict, 
         test_data_dirs: list, 
         model_path: Path, 
         save_dir_path: Path, 
         window_sizes: list, 
         logger: logging.Logger):
    """
    テストデータを用いてモデルの評価を行う関数
    Args:
        config (dict): 設定情報
        test_data_dirs (list): テストデータのディレクトリのリスト
        model_path (str): モデルのパス
        save_dir (str): 結果を保存するディレクトリ
        window_size (int): スライディングウィンドウのサイズ
        logger (logging.Logger): ロガー
    Returns:
        dict: マルチラベルの結果
        dict: スライディングウィンドウ解析の結果
    """
    # setup
    device, num_gpus = get_device_and_num_gpus()
    set_seed(42)
    
    # dataloaderの作成
    dataloader_factory = DataLoaderFactory(
        dataset_root=config.paths.dataset_root,
        batch_size=1,
        num_classes=config.test.num_classes,
        num_gpus=num_gpus
    )
    test_dataloaders = dataloader_factory.create_multi_label_test_dataloaders(test_data_dirs)
    
    # モデルのセットアップ
    model = setup_model(
        config=config, 
        device=device, 
        num_gpus=num_gpus, 
        mode='test', 
        model_path=model_path
    )
    
    # 推論
    inference = Inference(model, device)
    results = inference.run(save_dir_path, test_dataloaders)
    
    # 可視化
    visualizer = ResultsVisualizer(save_dir_path)
    # コンバーター
    converter = MultiToSingleLabelConverter(results)
    
    # マルチラベルを閾値でハードラベルに変換する手法の結果
    ## マルチラベルを閾値でハードラベルに変換
    hard_multi_label_results = converter.convert_soft_to_hard_multi_labels(threshold=0.5)
    converter.save_hard_multi_label_results(hard_multi_label_results, save_dir_path, methods = 'threshold_50%')

    # 正解ラベルの可視化
    visualizer.save_main_classes_visualization(hard_multi_label_results)
    # マルチラベルを閾値でハードラベルに変換した結果を可視化
    visualizer.save_multi_label_visualization(hard_multi_label_results,  methods = 'threshold_50%')


    ## 混同行列の計算
    calculator = ClassificationMetricsCalculator(num_classes=config.test.num_classes)
    video_metrics = calculator.calculate_multi_label_metrics_per_video(hard_multi_label_results)
    overall_metrics = calculator.calculate_multi_label_overall_metrics(hard_multi_label_results)
    ## 各動画フォルダにマルチラベルのメトリクスを保存
    save_video_metrics_to_csv(video_metrics, save_dir_path, methods = 'threshold_50%')
    ## 全体のメトリクスを保存
    save_overall_metrics_to_csv(overall_metrics, save_dir_path / 'fold_results', methods = 'threshold_50%')
    
    # スライディングウィンドウ解析
    analyzer = Analyzer(save_dir_path, config.test.num_classes)

    # スライディングウィンドウ解析の結果
    all_window_results = analyzer.analyze_sliding_windows(
        hard_multi_label_results,
        visualizer,
        calculator,
        window_sizes=window_sizes
    )

    # スライディングウィンドウ解析の結果を保存
    for window_key, sliding_window_results in all_window_results.items():
        # 結果を可視化
        visualizer.save_single_label_visualization(
            sliding_window_results,
            methods=window_key
        )
        # メトリクスを計算
        sliding_window_video_metrics = calculator.calculate_single_label_metrics_per_video(sliding_window_results)
        sliding_window_overall_metrics = calculator.calculate_single_label_overall_metrics(sliding_window_results)
        
        # 各動画フォルダにメトリクスを保存
        save_video_metrics_to_csv(
            sliding_window_video_metrics,
            save_dir_path,
            methods=window_key
        )
        
        # 全体のメトリクスを保存
        save_overall_metrics_to_csv(
            sliding_window_overall_metrics, 
            save_dir_path / 'fold_results',
            methods=window_key
        )
    
    return hard_multi_label_results, all_window_results
    

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-class organ classification model using ResNet')
    parser.add_argument('-c', '--config', help='Path to config file', default='config.yaml')
    return parser.parse_args()

def main():
    # 引数の解析
    args = parse_args()
    config = load_test_config(args.config)
    
    # 結果保存フォルダを作成
    Path(config.paths.save_dir).mkdir(exist_ok=True)
    
    # dataloaderの作成
    splitter = CrossValidationSplitter(splits=config.splits.root)
    split_folders = splitter.get_split_folders()
        
    # 全foldの結果を保存する辞書
    all_folds_hard_multi_label_results = {}
    all_folds_all_window_results = {}
    
    # window_sizeの探索範囲
    # window_sizes = list(range(9, 13, 2))
    window_sizes = [1, 11]
    
    # 全foldの結果を保存する辞書
    all_folds_hard_multi_label_results = {}
    all_folds_all_window_results = WindowSizeKey.initialize_results(window_sizes)
    
    for fold_idx, (split_data, model_path) in enumerate(zip(split_folders, config.paths.model_paths)):
        # fold用の結果保存フォルダを作成
        fold_save_dir_path = Path(config.paths.save_dir) / f"fold_{fold_idx}"
        fold_save_dir_path.mkdir(parents=True, exist_ok=True)

        # 各foldで独立したロガーを設定
        logger = setup_logging(fold_save_dir_path, f'test_fold_{fold_idx}')
        logger.info(f"Started test for fold {fold_idx}")

        hard_multi_label_results, all_window_results = test(config=config, 
                                                            test_data_dirs=split_data['test'],
                                                            model_path=model_path,
                                                            save_dir_path=fold_save_dir_path,
                                                            window_sizes=window_sizes,
                                                            logger=logger
                                                            )
        
        # 各foldの結果を全体の辞書に追加
        for folder_name, result in hard_multi_label_results.items():
            if folder_name not in all_folds_hard_multi_label_results:
                all_folds_hard_multi_label_results[folder_name] = result

        # Then store the results
        for window_key, sliding_window_results in all_window_results.items():
            for folder_name, result in sliding_window_results.items():
                all_folds_all_window_results[window_key][folder_name] = result

    # 全foldの結果を集約して評価指標を計算
    calculator = ClassificationMetricsCalculator(num_classes=config.test.num_classes)

    # 各window_sizeの全foldの結果を集約して評価指標を計算
    for window_key, all_window_results in all_folds_all_window_results.items():
        save_dir = Path(config.paths.save_dir) / window_key
        save_dir.mkdir(parents=True, exist_ok=True)
        metrics = calculator.calculate_all_folds_metrics(all_window_results, save_dir)
        

if __name__ == '__main__':
    main()