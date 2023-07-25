#!/usr/bin/env python3

import torch
import torch.nn as nn


class TNET(nn.Module):
    def __init__(self, fc_size):
        super(TNET, self).__init__()

        # Convolution layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=128,
                      kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

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
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
