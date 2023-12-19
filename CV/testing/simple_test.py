import torch

from ..lib.models.vgg import VGG

image = torch.rand(1,1,256,256)

net = VGG(spatial_dimensions=2)

print(net(image))