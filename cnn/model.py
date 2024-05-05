"""
Implementation of ResNet model with Dropout for stronger regularization
"""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout_rate=0.5):
        super().__init__()
        # first conv2d block + batch normalization + relu
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,  # 3x3 kernel
                padding=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        # second conv2d block + batch normalization
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=3,  # 3x3 kernel
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_ch),
            nn.Dropout(p=dropout_rate),
        )
        self.relu = nn.ReLU()
        # periodically double # of filters and downsample spatially
        # see page 4, section 3.3 residual networks in ResNet paper
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_ch),
        )
        # make parameters accessible
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride

    def forward(self, x):
        # F(x)
        out = self.conv1(x)
        out = self.conv2(out)

        # check if downsampling; if yes, run 1x1 conv2d to fix residual dims, otherwise just add
        if self.in_ch != self.out_ch or self.stride != 1:
            out += self.downsample(x)
        else:
            out += x

        # F(x) + x
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, layers, num_classes, dropout_rate):
        super().__init__()
        # ResNet 3.4 - "adopt batch normalization (BN) [16] right after each convolution and before activation"

        self.in_ch = 64  # starting filter depth; this value is updated with _make_layer

        # 7x7, 64, stride 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                padding=3,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 3x3, stride 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # residual block layers
        self.layer_0 = self._make_layer(64, layers[0], dropout_rate, start_stride=1)
        self.layer_1 = self._make_layer(128, layers[1], dropout_rate, start_stride=2)
        self.layer_2 = self._make_layer(256, layers[2], dropout_rate, start_stride=2)
        self.layer_3 = self._make_layer(512, layers[3], dropout_rate, start_stride=2)
        #
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)
        #

    def _make_layer(self, out_ch, block_depth, dropout_rate=0.5, start_stride=1):
        # on first section of new blocks (filter depth doubled), double stride to match dims, then use stride=1
        strides = [start_stride] + [1] * (block_depth - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_ch, out_ch, stride, dropout_rate))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer_0(out)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.avgpool(out)
        # reshape
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out
