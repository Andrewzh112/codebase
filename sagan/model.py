from torch import nn
import torch
from networks.layers import (ConvNormAct, SN_Linear, SN_Embedding,
                             SN_Conv2d, SN_ConvTranspose2d, SA_Conv2d)
from networks.utils import initialize_modules
from sagan.layers import ConditionalConvBNAct, CategoricalConditionalBatchNorm2d


class Discriminator(nn.Module):
    def __init__(self, img_channels, h_dim, img_size, num_classes=0):
        super().__init__()
        self.disc = nn.ModuleList([
            ConditionalConvBNAct(img_channels, h_dim, 'sn', 'down', activation='lrelu', normalization='bn'),
            SA_Conv2d(h_dim),
            ConditionalConvBNAct(h_dim, h_dim*2, 'sn', 'down', activation='lrelu', normalization='bn'),
            SA_Conv2d(h_dim*2),
            ConditionalConvBNAct(h_dim*2, h_dim*4, 'sn', 'down', activation='lrelu', normalization='bn'),
            ConditionalConvBNAct(h_dim*4, h_dim*8, 'sn', 'down', activation='lrelu', normalization='bn'),
            ConditionalConvBNAct(h_dim*8, h_dim*8, 'sn', 'down', activation='lrelu', normalization='bn'),
            ConditionalConvBNAct(h_dim*8, h_dim*8, 'sn', 'down', activation='lrelu', normalization='bn'),
            nn.AdaptiveAvgPool2d(1),
        ])
        self.in_features = h_dim*8
        self.fc = SN_Linear(in_features=self.in_features, out_features=1)
        if num_classes > 0:
            self.embed = SN_Embedding(num_embeddings=num_classes, embedding_dim=self.in_features)
        initialize_modules(self, init_type='ortho')

    def forward(self, x, y=None):
        for layer in self.disc:
            if isinstance(layer, (CategoricalConditionalBatchNorm2d, ConditionalConvBNAct)) and y is not None:
                x = layer(x, y)
            else:
                x = layer(x)
        h = x.squeeze()
        output = self.fc(h)
        if y is not None:
            output += torch.sum(self.embed(y) * h, dim=1, keepdim=True)
        return output

class Generator(nn.Module):
    def __init__(self, h_dim, z_dim, img_channels, img_size, num_classes=0):
        super().__init__()
        self.min_hw = (img_size // (2 ** 6))
        self.h_dim = h_dim
        self.project = SN_Linear(in_features=z_dim, out_features=h_dim*8 * self.min_hw**2, bias=False)
        self.gen = nn.ModuleList([
            CategoricalConditionalBatchNorm2d(num_classes=num_classes, num_features=h_dim*8, momentum=0.9),
            nn.ReLU(),
            ConditionalConvBNAct(h_dim*8, h_dim*8, 'sn', 'up', activation='relu', normalization='bn', num_classes=num_classes),
            ConditionalConvBNAct(h_dim*8, h_dim*8, 'sn', 'up', activation='relu', normalization='bn', num_classes=num_classes),
            ConditionalConvBNAct(h_dim*8, h_dim*4, 'sn', 'up', activation='relu', normalization='bn', num_classes=num_classes),
            ConditionalConvBNAct(h_dim*4, h_dim*2, 'sn', 'up', activation='relu', normalization='bn', num_classes=num_classes),
            SA_Conv2d(h_dim*2),
            ConditionalConvBNAct(h_dim*2, h_dim, 'sn', 'up', activation='relu', normalization='bn', num_classes=num_classes),
            SA_Conv2d(h_dim),
            SN_ConvTranspose2d(in_channels=h_dim, out_channels=img_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh()
        ])
        initialize_modules(self, init_type='ortho')

    def forward(self, x, y=None):
        batch_size = x.size(0)
        x = self.project(x)
        x_hat = x.view(
            batch_size,
            self.h_dim*8,
            self.min_hw,
            self.min_hw)
        for layer in self.gen:
            if isinstance(layer, (CategoricalConditionalBatchNorm2d, ConditionalConvBNAct)) and y is not None:
                x_hat = layer(x_hat, y)
            else:
                x_hat = layer(x_hat)
        return x_hat
