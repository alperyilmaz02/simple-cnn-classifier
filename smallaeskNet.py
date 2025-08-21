#!/usr/bin/env python3

import torch
import torch.nn as nn

class SmallAeskNet(nn.Module):
    def __init__(self, output_size=2, adaptive_pool_size=(2, 2), input_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 10x10 -> 5x5
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptive_pool_size)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 10, 10)
            out = self.forward_features(dummy)
            n_features = out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_size)
        )

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
