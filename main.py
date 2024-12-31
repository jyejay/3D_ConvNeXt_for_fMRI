import os
import torch
import random
import numpy as np
import argparse
from pathlib import Path

from models.convnext import ConvNeXt3D
from data.task_configs import TASK_CONFIGS, ROOT_DIR
from data.dataset import HCPDataset
from training.trainer import train_fold, create_folds

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Train HCP Classification Model')
    
    # Data arguments
    parser.add_argument('--task', type=str, required=True, choices=list(TASK_CONFIGS.keys()), 
                        help='Task to train on')
    
    # Model arguments
    parser.add_argument('--model_depths', type=int, nargs=4, default=[3, 3, 9, 3],
                        help='Depth of each stage')
    parser.add_argument('--model_dims', type=int, nargs=4, default=[96, 192, 384, 768],
                        help='Dimensions of each stage')
    parser.add_argument('--drop_path_rate', type=float, default=0.,
                        help='Drop path rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--project_name', type=str, default='hcp_classification')
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data and create folds
    subject_ids = []
    for root, _, files in os.walk(os.path.join(ROOT_DIR, args.task)):
        for file in files:
            if file.endswith('.npy'):
                subject_id = root.split(os.sep)[-3]
                if subject_id not in subject_ids:
                    subject_ids.append(subject_id)
    
    folds = create_folds(subject_ids, n_splits=5, random_state=args.seed)
    config = vars(args)  # Convert args to dictionary
    
    # Cross-validation training
    fold_accuracies = []
    for fold_idx, test_subjects in enumerate(folds):
        print(f"\nTraining Fold {fold_idx + 1}/5")
        
        # Prepare train/test split for this fold
        train_subjects = [subj for subj in subject_ids if subj not in test_subjects]
        
        train_dataset = HCPDataset(args.task, train_subjects)
        test_dataset = HCPDataset(args.task, test_subjects)

        # Create model for this fold
        model = ConvNeXt3D(
            in_chans=1,
            num_classes=TASK_CONFIGS[args.task]['num_classes'],
            depths=args.model_depths,
            dims=args.model_dims,
            drop_path_rate=args.drop_path_rate
        ).to(args.device)
        
        # Train model
        best_acc = train_fold(fold_idx, model, train_dataset, test_dataset, config)
        fold_accuracies.append(best_acc)
        
        print(f"Fold {fold_idx + 1} Best Accuracy: {best_acc:.2f}%")
    
    # Print final results
    print("\nCross-validation Results:")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%")
    print(f"Individual Fold Accuracies: {[f'{acc:.2f}%' for acc in fold_accuracies]}")

if __name__ == '__main__':
    main()