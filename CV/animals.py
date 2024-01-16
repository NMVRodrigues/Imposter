import os
import sys
import torch
import pandas as pd
import numpy as np
from torchmetrics import Accuracy, FBetaScore, MetricCollection

from rich.progress import track
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from lib.models.vgg import *
from datasets import ClassificationDataset

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.generate_metrics import cv_metrics
from utils.checkpointing import checkpoint, resume

validation_split = .2
shuffle_dataset = True
seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 1e-4

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

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


# Define a torchmetrics MetricCollection dict of accuracy and fbetascore
metrics = cv_metrics(n_classes=10, device=device)


NUM_ACCUMULATION_STEPS = 1

BEST_METRIC = 0

# Train the model
for epoch in range(25):
    running_loss = 0.0
    metrics['train'].reset()
    metrics['val'].reset()
    # Make sure gradient tracking is on
    model.train(True)

    for idx, (images, labels) in track(enumerate(train_loader), description=f'Training epoch {epoch}',
                                       total=len(train_loader)):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        running_loss += loss / NUM_ACCUMULATION_STEPS

        metrics_train = metrics['train'](outputs, labels)

        loss.backward()

        if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0) or (idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for idx, (images, labels) in track(enumerate(validation_loader), description=f'Validating...', total=len(validation_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss = loss_fn(outputs, labels)
            metrics_train = metrics['val'](outputs, labels)

    print(
        "Epoch {}    - Loss: {:.4f} - Accuracy: {:.4f} - F1: {:.4f} - Pr: {:.4f} - Rc: {:.4f} - AUC: {:.4f} - CalErr: {:.4f}".format(
            epoch,
            running_loss / len(train_loader),
            metrics['train']['Acc'].compute(),
            metrics['train']['F1'].compute(),
            metrics['train']['Pr'].compute(),
            metrics['train']['Rc'].compute(),
            metrics['train']['AUC'].compute(),
            metrics['train']['CalErr'].compute(),
        ))

    print(
        "Validation - Loss: {:.4f} - Accuracy: {:.4f} - F1: {:.4f} - Pr: {:.4f} - Rc: {:.4f} - AUC: {:.4f} - CalErr: {:.4f}".format(
            val_loss,
            metrics['val']['Acc'].compute(),
            metrics['val']['F1'].compute(),
            metrics['val']['Pr'].compute(),
            metrics['val']['Rc'].compute(),
            metrics['val']['AUC'].compute(),
            metrics['val']['CalErr'].compute(),
        ))

    # convnext-0.00001_fold1_best_epoch=6_V_AUC=0.695
    checkpoint(model, os.path.join('checkpoints', f"vgg-{float(lr)}-epoch={epoch}.pth"))
    if metrics['val']['AUC'].compute() > BEST_METRIC:
        BEST_METRIC = metrics['val']['AUC'].compute()
        checkpoint(model, os.path.join('checkpoints', f"vgg-{float(lr)}-best_epoch={epoch}_V_AUC={BEST_METRIC:.4f}.pth"))
