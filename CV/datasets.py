from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image, ImageReadMode
from PIL import Image


class ClassificationDataset(Dataset):
    def __init__(self, annotations_file, label_encoding=False, transform=None, target_transform=None):
        self.data = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
        self.label_encoding = label_encoding

        if self.label_encoding is True:
            self.mapping = {elem: i for i, elem in enumerate(list(set(self.data['class'])))}
            self.data['class'] = self.data['class'].map(self.mapping)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data['image'][idx]
        image = read_image(img_path, mode=ImageReadMode.RGB)
        label = self.data['class'][idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")