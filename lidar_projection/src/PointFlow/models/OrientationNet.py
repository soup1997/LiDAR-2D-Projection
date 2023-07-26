import torch
import torch.nn as nn

from .SqueezeNet import *


class ONET(nn.Module):
    def __init__(self):
        super(ONET, self).__init__()
        self.squeezenet = SqueezeNet()

    def forward(self, x):
        x = self.squeezenet(x)
        x = x.view(x.size(0), -1)

        return x