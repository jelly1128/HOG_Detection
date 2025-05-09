import torch
import torch.nn as nn
import torchvision.models as models


class SingleLabelClassificationModel(nn.Module):
    """
    シングルラベル分類用のモデル。
    model_architectureでResNet/EfficientNet等を選択可能。
    """
    def __init__(self, num_classes, model_architecture='resnet18', pretrained=False, freeze_backbone=False):
        super(SingleLabelClassificationModel, self).__init__()
        self.num_classes = num_classes
        # アーキテクチャごとに初期化
        if model_architecture == 'resnet18':
            self.network = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif model_architecture == 'resnet34':
            self.network = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif model_architecture == 'resnet50':
            self.network = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif model_architecture == 'resnet101':
            self.network = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif model_architecture == 'resnet152':
            self.network = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif model_architecture == 'efficientnet-b0':
            self.network = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.classifier[1].in_features
            self.network.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model architecture: {model_architecture}")
        if freeze_backbone:
            for param in self.network.parameters():
                param.requires_grad = False
        self.output_layer = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.network(x)
        out = self.output_layer(features)
        return out


class MultiTaskDetectionModel(nn.Module):
    def __init__(self, num_classes=6, model_architecture='resnet18', pretrained=False, freeze_backbone=False):
        """
        Args:
            num_classes: 全クラス数（例: 6, 7, 15）
            pretrained: ImageNetの重みを使用するかどうか
            freeze_backbone: バックボーンのパラメータを固定するかどうか
        """
        super(MultiTaskDetectionModel, self).__init__()
        
        # ResNet18の初期化（pretrainedの場合はImageNetの重みを利用）
        if model_architecture == 'resnet18':
            self.network = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif model_architecture == 'resnet34':
            self.network = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif model_architecture == 'resnet50':
            self.network = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif model_architecture == 'resnet101':
            self.network = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif model_architecture == 'resnet152':
            self.network = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif model_architecture == 'efficientnet-b0':
            self.network = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.classifier[1].in_features
            self.network.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model architecture: {model_architecture}")

        # 共通の特徴次元を取得した後の処理
        if freeze_backbone:
            for param in self.network.parameters():
                param.requires_grad = False
        
        # クラス数に応じて出力層を設定
        if num_classes == 6:
            # 6クラスの場合は主クラスのみ
            self.main_head = nn.Linear(feature_dim, 6)
            self.unclear_head = None
        elif num_classes == 7:
            # 7クラスの場合は主クラス6 + 不鮮明クラス1
            self.main_head = nn.Linear(feature_dim, 6)
            self.unclear_head = nn.Linear(feature_dim, 1)
        elif num_classes == 15:
            # 15クラスの場合は主クラス6 + 不鮮明クラス9
            self.main_head = nn.Linear(feature_dim, 6)
            self.unclear_head = nn.Linear(feature_dim, 9)
        else:
            raise ValueError(f"Unsupported number of classes: {num_classes}")
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力テンソル (batch_size, c, h, w)
        Returns:
            連結された出力テンソル
            - 6クラスの場合: (batch_size, 6)
            - 7クラスの場合: (batch_size, 7)
            - 15クラスの場合: (batch_size, 15)
        """
        # バックボーンによる特徴抽出
        features = self.network(x)  # shape: (batch_size, feature_dim)
        
        # 主クラスの出力
        main_out = self.main_head(features)
        
        # クラス数に応じて出力を返す
        if self.num_classes == 6:
            return main_out
        else:
            # 不鮮明クラスの出力と連結
            unclear_out = self.unclear_head(features)
            return torch.cat([main_out, unclear_out], dim=1)

# ======================================
# モデルのインスタンス生成と出力例
# ======================================

def main():
    # 例: 主クラス6、かつ不鮮明クラス9 → 合計15クラスとして出力
    model = MultiTaskDetectionModel(num_classes=15, pretrained=True, freeze_backbone=False)

    # ダミー入力例 (batch_size=4, RGB画像 224x224)
    dummy_input = torch.randn(4, 3, 224, 224)

    # 順伝播
    output = model(dummy_input)  # 出力 shape: (4, 15)
    print("Concatenated output shape:", output.shape)

if __name__ == '__main__':
    main()


