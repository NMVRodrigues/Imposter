import torch

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
                 first_depth: int,
                 size: int = 2,
                 dimension: int = 2,
                 ):

        """
            Args:
                input_channels (List[int]): list of input channels for convolutions.
                first_depth (int): number of output channels for the first convolution.
                size (int, optional): number of convolution operation
                dimension (int, optional): If the block is going to be defined in 2D or 3D, defaults to 2D.
        """

        super().__init__()
        self.input_channels = input_channels
        self.first_depth = first_depth
        self.size = size
        self.dimension = dimension
        self.layers = torch.nn.ModuleList([])

        if self.dimension == 2:
            for i in range(size):
                if i == 0:
                    self.layers.append(torch.nn.Conv2d(input_channels, first_depth, 3, padding=1))
                    self.layers.append(torch.nn.GELU())
                    self.layers.append(torch.nn.BatchNorm2d(first_depth))
                else:
                    self.layers.append(torch.nn.Conv2d(input_channels, first_depth*2, 3, padding=1))
                    self.layers.append(torch.nn.GELU())
                    self.layers.append(torch.nn.BatchNorm2d(first_depth*2))

        else:
            for i in range(size):
                if i == 0:
                    self.layers.append(torch.nn.Conv3d(input_channels, first_depth, 3, padding=1))
                    self.layers.append(torch.nn.GELU())
                    self.layers.append(torch.nn.BatchNorm3d(first_depth))
                else:
                    self.layers.append(torch.nn.Conv3d(input_channels, first_depth*2, 3, padding=1))
                    self.layers.append(torch.nn.GELU())
                    self.layers.append(torch.nn.BatchNorm3d(first_depth*2))


    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

