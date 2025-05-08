from pathlib import Path
import matplotlib.pyplot as plt
import csv

class TrainingMonitor:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    def plot_learning_curve(self, loss_history: dict):
        epochs = range(1, len(loss_history['train']) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss_history['train'], label='Train Loss')
        plt.plot(epochs, loss_history['val'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Learning Curve')
        plot_path = Path(self.save_dir, 'learning_curve.png')
        
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Learning curve saved to {plot_path}")
    
    def save_loss_to_csv(self, loss_history: dict):
        csv_path = Path(self.save_dir, 'loss_history.csv')
        
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
            
            for epoch, (train_loss, val_loss) in enumerate(
                zip(loss_history['train'], loss_history['val']), start=1
            ):
                writer.writerow([epoch, train_loss, val_loss])
                
        print(f"Loss history saved to {csv_path}")