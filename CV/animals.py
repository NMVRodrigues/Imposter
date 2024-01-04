import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from rich.progress import track
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from lib.models.vgg import *
from datasets import ClassificationDataset

validation_split = .2
shuffle_dataset = True
seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((32,32), antialias=True),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ClassificationDataset(os.path.join('../Datasets', 'CV', 'animals10', 'annotations.csv'), label_encoding=True, transform=transform)

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=valid_sampler)

model = vgg16(n_channels=3, n_classes=10)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Initialize the accuracy metric


# Define the number of accumulation steps

# Train the model
for epoch in track(range(10), description='Training...'):
    running_loss = 0.0
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        #accuracy.update(outputs.cpu(), labels.cpu())


    print(
        "Epoch {} - Loss: {:.4f} - Accuracy: {:.4f}".format(epoch, loss.item(), 0))
    #accuracy.reset()