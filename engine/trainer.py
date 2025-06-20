from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, device, is_multilabel=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.is_multilabel = is_multilabel

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for images, _, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)

            if self.is_multilabel:
                # モデルの出力次元数を確認
                if outputs.shape[1] > 6:  # マルチタスクモデルの場合
                    main_outputs, unclear_outputs = outputs[:, :6], outputs[:, 6:]
                    main_labels, unclear_labels = labels[:, :6], labels[:, 6:] 
                    main_loss = self.criterion(main_outputs, main_labels)
                    unclear_loss = self.criterion(unclear_outputs, unclear_labels)
                    loss = main_loss + unclear_loss
                else:  # 6クラスのみの場合
                    loss = self.criterion(outputs, labels)
            else:
                # シングルラベル（整数ラベル）用
                loss = self.criterion(outputs, labels.long())

            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        
        return total_loss / len(train_loader)