import os
import torch
import numpy as np
from torch.utils.data import Dataset
from .task_configs import TASK_CONFIGS, ROOT_DIR


class HCPDataset(Dataset):
    """
    HCP Dataset for different tasks
    
    Args:
        task (str): Task name ('WM', 'MOTOR', etc.)
        subject_ids (list): List of subject IDs to include
        transform (callable, optional): Optional transform to be applied on a sample
    """
    def __init__(self, task, subject_ids, transform=None):
        self.file_paths = []
        self.labels = []
        self.transform = transform
        
        if task not in TASK_CONFIGS:
            raise ValueError(f"Task {task} not supported. Available tasks: {list(TASK_CONFIGS.keys())}")
        
        label_mapping = TASK_CONFIGS[task]['label_mapping']
        
        # 데이터 파일 경로와 레이블 수집
        for subject_id in subject_ids:
            subject_dir = os.path.join(ROOT_DIR, task, subject_id)
            for root, _, files in os.walk(subject_dir):
                for file in files:
                    if file.endswith('.npy'):
                        file_path = os.path.join(root, file)
                        self.file_paths.append(file_path)
                        for label_str, label_num in label_mapping.items():
                            if label_str in file_path:
                                self.labels.append(label_num)
                                break

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # 데이터 로드
        data = np.load(file_path)
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        
        # 변환 적용 (필요한 경우)
        if self.transform:
            data_tensor = self.transform(data_tensor)
            
        label_tensor = torch.tensor(label, dtype=torch.long)
        return data_tensor, label_tensor