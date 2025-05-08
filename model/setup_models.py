import torch
import torch.nn as nn
from .cnn_models import MultiLabelDetectionModel, MultiTaskDetectionModel


def setup_model(config, device, num_gpus, mode='train', model_path=None):
    """
    モデルをセットアップし、GPUに移動し、並列化する。

    Args:
        config: 設定オブジェクト（config.training.num_classes などが必要）
        device: 使用するデバイス（'cuda' または 'cpu'）
        num_gpus: 使用するGPUの数
        mode: 'train' または 'test'（デフォルトは 'train'）
        model_path: テスト時に読み込むモデルのパス（mode='test' の場合に必須）

    Returns:
        model: セットアップされたモデル
    """
    # モデルの初期化
    model_type = config.training.model_type if mode == 'train' else config.test.model_type
    if model_type == 'multitask':
        model = MultiTaskDetectionModel(
            num_classes=config.training.num_classes if mode == 'train' else config.test.num_classes,  # テスト時はnum_classes=config.testing.num_classes
            model_architecture=config.training.model_architecture if mode == 'train' else config.test.model_architecture,  # テスト時はmodel_architecture=config.testing.model_architecture
            pretrained=config.training.pretrained if mode == 'train' else False,                      # テスト時はpretrained=False
            freeze_backbone=config.training.freeze_backbone if mode == 'train' else False             # テスト時はfreeze_backbone=False
        )
    else:
        raise ValueError(f"Invalid model type: {config.training.model_type}")

    # テスト時は学習済みの重みを読み込む
    if mode == 'test':
        if model_path is None:
            raise ValueError("model_path must be provided in test mode.")
        
        # モデルの重みを読み込む
        state_dict = torch.load(model_path)
        
        # DataParallelで保存されたモデルの重みを処理
        if 'module.' in list(state_dict.keys())[0]:
            if num_gpus > 1:
                model = nn.DataParallel(model)
            model.load_state_dict(state_dict)
        else:
            if num_gpus > 1:
                model = nn.DataParallel(model)
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
    else:
        # 学習時はDataParallelを適用
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for training")
            model = nn.DataParallel(model)

    # モデルをデバイスに移動
    model = model.to(device)
    
    return model