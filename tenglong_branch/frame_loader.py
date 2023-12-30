import os
import torch
import torchvision


class RealVideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = []

        for folder in os.listdir(root):
            if not os.path.isdir(os.path.join(root, folder)):
                continue
            folder_id = int(folder)
            for filename in os.listdir(os.path.join(root, folder)):
                if filename.endswith('jpg') or filename.endswith('png'):
                    filepath = os.path.join(root, folder, filename)
                    self.data.append((filepath, folder_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filepath, folder_id = self.data[idx]
        try:
            image = torchvision.io.read_image(filepath).float() / 255.0
        except:
            print(filepath)
            raise
        if self.transform:
            image = self.transform(image)
        return image, folder_id
    

class FakeVideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = []

        for folder in os.listdir(root):
            if not os.path.isdir(os.path.join(root, folder)):
                continue
            for filename in os.listdir(os.path.join(root, folder)):
                if filename.endswith('jpg') or filename.endswith('png'):
                    filepath = os.path.join(root, folder, filename)
                    self.data.append(filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filepath = self.data[idx]
        image = torchvision.io.read_image(filepath).float() / 255.0
        if self.transform:
            image = self.transform(image)
        return image


if __name__ == '__main__':
    # root = '/home/longteng/datasets/ff++23/tlong_all_frames/original/train'
    root = '/home/longteng/datasets/ff++23/tlong_all_frames/neuraltextures'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224), antialias=True),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # dataset = RealVideoFrameDataset(root, transform)
    dataset = FakeVideoFrameDataset(root, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    for i, images in enumerate(dataloader):
        print(i, images.shape)

    
