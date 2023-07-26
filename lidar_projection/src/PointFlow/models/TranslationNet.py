#!/usr/bin/env python3

import torch
import torch.nn as nn

from .FlowNet2S import *


class TNET(nn.Module):
    def __init__(self, fc_size):
        super(TNET, self).__init__()

        self.flownet2 = FlowNet2S(batchNorm=False, div_flow=20)

        # Fully Connected layers
        self.fc = nn.Sequential(
            nn.Linear(fc_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(16, 3))

    def forward(self, x):
        x = self.flownet2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
