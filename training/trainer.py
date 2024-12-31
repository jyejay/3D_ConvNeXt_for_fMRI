import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from data.dataset import HCPDataset
from data.task_configs import TASK_CONFIGS
import os

class Trainer:
    def __init__(self, model, optimizer, criterion, device, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        
    def train_one_epoch(self, data_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in data_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        return epoch_loss, epoch_acc

    def evaluate(self, data_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        return epoch_loss, epoch_acc

def train_fold(fold_idx, model, train_dataset, test_dataset, config):
    """Train and evaluate model on a single fold"""
    wandb.init(
        project=config['project_name'],
        name=f"fold_{fold_idx}",
        config=config,
        reinit=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=config['learning_rate']),
        criterion=torch.nn.CrossEntropyLoss(),
        device=config['device'],
        config=config
    )
    
    best_acc = 0
    
    for epoch in tqdm(range(config['num_epochs']), desc=f"Fold {fold_idx} Training"):
        train_loss, train_acc = trainer.train_one_epoch(train_loader)
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint_name = f"task_{config['task']}_depths{''.join(map(str, config['model_depths']))}_dims{''.join(map(str, config['model_dims']))}_fold{fold_idx}_best.pth"
            checkpoint_path = os.path.join(config['checkpoint_dir'], checkpoint_name)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_acc': best_acc,
                'task': config['task'],
                'model_depths': config['model_depths'],
                'model_dims': config['model_dims'],
                'config': config  # 전체 설정도 저장
            }, checkpoint_path)
        
        
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'best_acc': best_acc
        })
    
    wandb.finish()
    return best_acc

def create_folds(subject_ids, n_splits=5, random_state=42):
    """Create stratified folds for cross-validation"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for _, test_idx in kf.split(subject_ids):
        folds.append([subject_ids[i] for i in test_idx])
    return folds