#!/usr/bin/env python3

import torch
import torch.nn as nn


class AeskNet(nn.Module):
    def __init__(self, output_size=2, adaptive_pool_size=(4, 4), input_channels=3):
        """
        output_size: number of classes
        adaptive_pool_size: output size of adaptive pooling before fc layers
        input_channels: number of channels in the input (default 3 for RGB)
        """
        super(AeskNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptive_pool_size)
        self.adaptive_pool_size = adaptive_pool_size

        # ---- Fully Connected Layers ----
        # Compute input features dynamically using a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 64, 64)  # arbitrary size
            out = self._forward_conv(dummy)
            out = self.adaptive_pool(out)
            n_features = out.view(1, -1).size(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, output_size)
        )

    def _forward_conv(self, x):
        """ Forward only through conv layers (helper function) """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layers(x)
        return x
