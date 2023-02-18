import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomAffine, Normalize
from networks import Lenet
import torch.cuda as cuda
import pytorch_lightning as pl
from pl import ClsNet
from pytorch_lightning.callbacks import RichProgressBar


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

train_loader = DataLoader(train_data, 128, shuffle=True, num_workers=0, pin_memory=True)

test_loader = DataLoader(test_data, 128, shuffle=False, num_workers=0, pin_memory=True)

device = 'cuda' if cuda.is_available() else 'cpu'

net = Lenet(dims=(24, 24, 1)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)

# Define the Lightning trainer
trainer = pl.Trainer(gpus=1, max_epochs=10, accumulate_grad_batches=4, callbacks=[RichProgressBar()])

# Call the PL module
model = ClsNet(net, optimizer, loss_fn, 10)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
