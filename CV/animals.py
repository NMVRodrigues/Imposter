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
    transforms.Resize((128,128), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ClassificationDataset(os.path.join('..\\Datasets', 'CV', 'animals10', 'annotations.csv'), label_encoding=True, transform=transform)

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

model = vgg16(n_channels=3, n_classes=10).to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


# Define the number of accumulation steps

NUM_ACCUMULATION_STEPS = 1

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for idx, (images, labels) in track(enumerate(train_loader), description=f'Training epoch {epoch}', total=len(train_loader)):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss = loss / NUM_ACCUMULATION_STEPS

        predictions = torch.softmax(outputs,1)
        predicted_classes = torch.argmax(predictions, 1)

        accuracy = (predicted_classes == labels).sum().float() / float(labels.size(0))

        loss.backward()
        if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0) or (idx + 1 == len(train_loader)):
            # Update Optimizer
            optimizer.step()
            optimizer.zero_grad()

    print(
        "Epoch {} - Loss: {:.4f} - Accuracy: {:.4f}".format(epoch, loss.item(), accuracy))