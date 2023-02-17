import torch.nn as nn
import torch.nn.functional as F

class Lenet(nn.Module):

    def __init__(self, dims):
        super(Lenet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(dims[-1], 6, kernel_size=5, padding=2),
            adn_block(6, dim=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            adn_block(16, dim=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            adn_block(120, dim=1),
            nn.Linear(120, 84),
            adn_block(84, dim=1),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.classifier(x)
        return x

def adn_block(channels, dim, dropout=0.2):
    norm_layer = nn.BatchNorm2d(channels) if dim == 2 else nn.BatchNorm1d(channels)
    return nn.Sequential(
        nn.ReLU(),
        nn.Dropout(p=dropout),
        norm_layer
    )