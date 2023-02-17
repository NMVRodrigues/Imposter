import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomAffine, Normalize
from networks import Lenet
from torcheval.metrics.classification import MulticlassAccuracy
import torch.cuda as cuda

transforms = Compose([ToTensor(), RandomAffine(degrees=(30, 70)), Normalize((0.1307,), (0.3081,))])

train_data = torchvision.datasets.MNIST(
    root = 'data',
    train = True,
    transform = transforms,
    download = True,
)
test_data = torchvision.datasets.MNIST(
    root = 'data',
    train = False,
    transform = transforms
)

train_loader = DataLoader(train_data, 32, shuffle=True, num_workers=0, pin_memory=True)

test_loader = DataLoader(test_data, 32, shuffle=False, num_workers=0, pin_memory=True)

device = 'cuda' if cuda.is_available() else 'cpu'

model = Lenet(dims=(24,24,1)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Initialize the accuracy metric
accuracy = MulticlassAccuracy()

# Define the number of accumulation steps
accumulation_steps = 4

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Calculate gradients and accumulate them
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        accuracy.update(outputs.cpu(), labels.cpu())

        # Update the model parameters at the end of accumulation_steps batches
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    print(
        "Epoch {} - Loss: {:.4f} - Accuracy: {:.4f}".format(epoch, running_loss / len(train_data), accuracy.compute()))
    accuracy.reset()