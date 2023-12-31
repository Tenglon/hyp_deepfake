import os
import torch
from torch.utils.data import Dataset, DataLoader

class Fakedataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): 数据集的目录路径。
            transform (callable, optional): 应用于样本的可选变换。
        """
        self.root_dir = root_dir
        
        # 加载所有样本的路径和标签
        self.samples = []
        for label, category in enumerate(sorted(os.listdir(self.root_dir ))):
            category_dir = os.path.join(self.root_dir, category)
            for file in os.listdir(category_dir):
                if file.endswith('.pt'):
                    self.samples.append((os.path.join(category_dir, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path, label = self.samples[idx]
        feature = torch.load(feature_path).to(torch.float32).squeeze()
        
        return feature, label


class Ffplusplusc23Dataset(Dataset):
    def __init__(self, root_dir,train=True):
        """
        Args:
            root_dir (string): 数据集的目录路径。
            transform (callable, optional): 应用于样本的可选变换。
        """
        self.root_dir = root_dir
        
        if train:
            self.root_dir = os.path.join(self.root_dir, 'train')
        else:
            self.root_dir = os.path.join(self.root_dir, 'test')
        self.samples = []
        # 加载所有样本的路径和标签
        for label, category in enumerate(sorted(os.listdir(self.root_dir ))):
            category_dir = os.path.join(self.root_dir, category)
            for file in os.listdir(category_dir):
                if file.endswith('.pt'):
                    self.samples.append((os.path.join(category_dir, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path, label = self.samples[idx]
        feature = torch.load(feature_path).to(torch.float32).squeeze()

        return feature, label
    

def create_loaders(data_dir, batch_size: int, num_workers: int):

    train_set = Ffplusplusc23Dataset(
        root_dir=data_dir,
        train=True,
    )

    test_set = Ffplusplusc23Dataset(
        root_dir=data_dir,
        train=False,
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, test_loader