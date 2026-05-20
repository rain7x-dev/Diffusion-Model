import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.data = datasets.MNIST(root='./data', train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
