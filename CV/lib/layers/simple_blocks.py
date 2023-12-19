import torch

from ..activations import activation_factory

class GlobalPooling(torch.nn.Module):
    def __init__(self, mode: str = "max"):
        """Wrapper for average and maximum pooling

        Args:
            mode (str, optional): pooling mode. Can be one of "average" or
            "max". Defaults to "max".
        """
        super().__init__()
        self.mode = mode

        self.get_op()

    def get_op(self):
        if self.mode == "average":
            self.op = torch.mean
        elif self.mode == "max":
            self.op = torch.max
        else:
            raise "mode must be one of [average,max]"

    def forward(self, X):
        if len(X.shape) > 2:
            X = self.op(X.flatten(start_dim=2), -1)
            if self.mode == "max":
                X = X.values
        return X

class vgg_block(torch.nn.Module):
    """
    Implementation of a simple vgg 2d and 3d convolutional block
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 size: int = 2,
                 dimension: int = 2,
                 act: str = 'gelu',
                 ):

        """
            Args:
                input_channels (List[int]): list of input channels for convolutions.
                output_channels (int): number of output channels for the first convolution.
                size (int, optional): number of convolution operation
                dimension (int, optional): If the block is going to be defined in 2D or 3D, defaults to 2D.
                act (str, optional): Activation function to use, defaults to gelu.
        """

        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.size = size
        self.dimension = dimension
        self.layers = torch.nn.ModuleList([])

        self.act = activation_factory[act]

        if self.dimension == 2:
            for i in range(size):
                if i == 0:
                    self.layers.append(torch.nn.Conv2d(self.input_channels, self.output_channels, 3, padding=1))
                    self.layers.append(self.act)
                    self.layers.append(torch.nn.BatchNorm2d(self.output_channels))
                else:
                    self.layers.append(torch.nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1))
                    self.layers.append(self.act)
                    self.layers.append(torch.nn.BatchNorm2d(self.output_channels))
            self.layers.append(torch.nn.MaxPool2d(2, 2))

        else:
            for i in range(size):
                if i == 0:
                    self.layers.append(torch.nn.Conv3d(self.input_channels, self.output_channels, 3, padding=1))
                    self.layers.append(self.act)
                    self.layers.append(torch.nn.BatchNorm3d(self.output_channels))
                else:
                    self.layers.append(torch.nn.Conv3d(self.output_channels, self.output_channels, 3, padding=1))
                    self.layers.append(self.act)
                    self.layers.append(torch.nn.BatchNorm3d(self.output_channels * 2))
            self.layers.append(torch.nn.MaxPool3d(2, 2))


    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

