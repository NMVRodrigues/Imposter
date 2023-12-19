import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.models.vgg import *

# requires batch size for batchnorm to work
image = torch.rand(2,1,256,256)

net = vgg16()

print(net(image))