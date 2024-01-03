from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image


class ClassificationDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_data = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = self.img_data['image'][idx]
        image = read_image(img_path)
        label = self.img_data['class'][idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

