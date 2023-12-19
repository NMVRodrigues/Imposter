import torch

from typing import List

class Resnet(torch.nn.Module):
    """
    Simple ResNet model builder, supporting multiple ResNet sizes
    """
    def __init__(self,
                 spatial_dimensions: int = 3,
                 n_channels: int = 1,
                 n_classes: int = 2,
                 feature_extraction=None,
                 maxpool_structure=None,
                 adn_fn=None,
                 convolution_structure: List[int] = [64,128,256,512,512],
                 classification_structure: List[int] = [512, 512, 512],
                 ) -> torch.nn.Module :
        super().__init__()

        self.spatial_dimension = spatial_dimensions
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.feature_extraction = feature_extraction
        self.maxpool_structure = maxpool_structure
        self.adn_fn = adn_fn
        self.convolution_structure = convolution_structure
        self.classification_structure = classification_structure

        self.initial_conv = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, 64, 64, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        )

        