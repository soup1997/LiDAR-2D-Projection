#!/usr/bin/env python3

import torch
import torch.nn as nn

from .FlowNetS import *


class ONET(nn.Module):
    def __init__(self, fc_size):
        super(ONET, self).__init__()

        self.flownet = FlowNetS(batchNorm=False, input_channels=6)

        # Fully Connected layers
        self.fc = nn.Sequential(
            nn.Linear(fc_size, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 3),
            nn.LeakyReLU(0.1, inplace=True))


    def forward(self, x):
        x = self.flownet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
