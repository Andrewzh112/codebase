"""
https://github.com/crcrpar/pytorch.sngan_projection/tree/master/links
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from networks.layers import ConvNormAct


class ConditionalNorm(nn.Module):
    """https://github.com/rosinality/sagan-pytorch/blob/master/model_resnet.py"""
    def __init__(self, n_class, in_channel):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)
        self.embed = nn.Embedding(n_class, in_channel * 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.bn(input)
        embed = self.embed(class_id)
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta

        return out


class ConditionalConvBNAct(ConvNormAct):
    def __init__(self, in_channels, out_channels, conv_type='basic', mode=None,
                 activation='relu', normalization='bn', groups=1, kernel_size=None, num_classes=0):
        super().__init__(in_channels, out_channels, conv_type, mode,
                         activation, normalization, groups, kernel_size)
        if num_classes > 0:
            self.norm = ConditionalNorm(num_classes, out_channels)
        self.block = nn.ModuleList([self.conv, self.norm, self.act])

    def forward(self, x, y=None):
        for layer in self.block:
            if isinstance(layer, ConditionalNorm) and y is not None:
                x = layer(x, y)
            else:
                x = layer(x)
        return x
