import torchvision
import torch
import os
from torch.utils.data import Dataset, DataLoader

# from .path_config import data_dir



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
    
    def _normalize(self, feature):
        mean = torch.mean(feature)
        std = torch.std(feature)
        return (feature - mean) / std

    def __getitem__(self, idx):
        feature_path, label = self.samples[idx]
        feature = torch.load(feature_path).to(torch.float32)
        feature = self._normalize(feature)
        


        return feature, label

    


#  class Ffplusplusc23DatasetFactory:
#     train_set = Ffplusplusc23Dataset(
#         root=data_dir,
#         train=True,
      
#     )

#     test_set = Ffplusplusc23Dataset(
#         root=data_dir,
#         train=False,
    
#     )

#     @classmethod
#     def create_train_loaders(cls, batch_size: int):
#         train_loader = DataLoader(
      
#             batch_size=batch_size,
#             shuffle=True,
#         )

#         test_loader = DataLoader(
          
#             batch_size=batch_size,
#             shuffle=False,
#         )

#         return train_loader, test_loader

