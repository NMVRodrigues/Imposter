import torch

from typing import List

from ..layers.simple_blocks import GlobalPooling,  vgg_block
from ..layers.linear_blocks import MLP
from ..layers.adn import get_adn_fn


class VGG(torch.nn.Module):
    """
    Simple VGG model builder, supporting multiple VGG sizes
    """
    def __init__(self,
                 spatial_dimensions: int = 3,
                 n_channels: int = 1,
                 n_classes: int = 2,
                 feature_extraction=None,
                 maxpool_structure=None,
                 adn_fn=None,
                 channel_structure: List[int] = [64,128,256,512,512],
                 convolution_structure: List[int] = [2,2,3,3,3],
                 classification_structure: List[int] = [512, 512, 512],
                 ) -> torch.nn.Module :
        super().__init__()

        self.spatial_dimensions = spatial_dimensions
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channel_structure = channel_structure
        self.convolution_structure = convolution_structure
        self.feature_extraction = feature_extraction
        self.maxpool_structure = maxpool_structure
        self.adn_fn = adn_fn

        self.body = torch.nn.ModuleList([])
        self.channel_structure.insert(0, self.n_channels)

        for i in range(len(convolution_structure)):
            self.body.append(vgg_block(input_channels=self.channel_structure[i],
                                       output_channels=self.channel_structure[i + 1],
                                       size=self.convolution_structure[i],
                                       dimension=self.spatial_dimensions
                                       )
                             )

        if self.n_classes == 2:
            final_n = 1
            self.last_act = torch.nn.Sigmoid()
        else:
            final_n = self.n_classes
            self.last_act = torch.nn.Softmax(-1)

        self.classification_layer = torch.nn.Sequential(
            GlobalPooling(),
            MLP(
                channel_structure[-1],
                final_n,
                classification_structure,
                adn_fn=get_adn_fn(1, "batch", "gelu"),
            ),
        )

    def forward_features(
            self, x: torch.Tensor, batch_idx: int = None
    ) -> torch.Tensor:
        """Forward method for features only.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output (classification)
        """
        return self.forward(x, return_features=True)

    def forward(
            self,
            x: torch.Tensor,
            return_features: bool = False,
    ) -> torch.Tensor:
        """Forward method.

        Args:
            x (torch.Tensor): input tensor
            return_features (bool, optional): returns the features before
                applying classification layer. Defaults to False.

        Returns:
            torch.Tensor: output (classification)
        """
        for block in self.body:
            x = block.forward(x)
        if return_features is True:
            return x

        return self.classification_layer(x)
    

def _make_vgg(params):
    return VGG(**params)


def vgg16(spatial_dimensions: int = 2,
          n_channels: int = 1,
          n_classes: int = 2,
          feature_extraction = None,
          maxpool_structure = None,
          adn_fn = None
          ) -> VGG : 
    
    params = {
        'spatial_dimensions': spatial_dimensions,
        'n_channels': n_channels,
        'n_classes': n_classes,
        'feature_extraction' : None,
        'maxpool_structure' : None,
        'adn_fn' : None,
        'channel_structure': [64,128,256,512,512],
        'convolution_structure': [2,2,3,3,3],
        'classification_structure':[512, 512, 512],
    }

    return _make_vgg(params)