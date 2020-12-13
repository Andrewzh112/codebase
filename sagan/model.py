from torch import nn
from networks.layers import (ConvNormAct, SN_Linear,
                             SN_Conv2d, SN_ConvTranspose2d, SA_Conv2d)
from networks.utils import initialize_modules


class Discriminator(nn.Module):
    def __init__(self, img_channels, h_dim, img_size):
        super().__init__()
        self.disc = nn.Sequential(
            SN_Conv2d(in_channels=img_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
            ConvNormAct(h_dim, h_dim*2, 'sn', 'down', activation='lrelu', normalization='bn'),
            ConvNormAct(h_dim*2, h_dim*4, 'sn', 'down', activation='lrelu', normalization='bn'),
            ConvNormAct(h_dim*4, h_dim*8, 'sn', 'down', activation='lrelu', normalization='bn'),
            nn.AdaptiveAvgPool2d(1),
        )
        self.in_features = h_dim*8
        self.fc = SN_Linear(in_features=self.in_features, out_features=1)
        initialize_modules(self)

    def forward(self, x):
        x = self.disc(x)
        x = x.view(-1, self.in_features)
        return self.fc(x)


class Generator(nn.Module):
    def __init__(self, h_dim, z_dim, img_channels, img_size):
        super().__init__()
        self.min_hw = (img_size // (2 ** 4)) ** 2
        self.h_dim = h_dim
        self.project = SN_Linear(in_features=z_dim, out_features=h_dim*8 * self.min_hw ** 2, bias=False)
        self.gen = nn.Sequential(
            nn.BatchNorm2d(h_dim*8, momentum=0.9),
            nn.ReLU(),
            ConvNormAct(h_dim*8, h_dim*4, 'sn', 'up', activation='relu', normalization='bn'),
            ConvNormAct(h_dim*4, h_dim*2, 'sn', 'up', activation='relu', normalization='bn'),
            SA_Conv2d(h_dim*2),
            ConvNormAct(h_dim*2, h_dim, 'sn', 'up', activation='relu', normalization='bn'),
            SN_ConvTranspose2d(in_channels=h_dim, out_channels=img_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh()
        )
        initialize_modules(self)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.project(x)
        x = x.view(batch_size, self.h_dim*8, self.min_hw, self.min_hw)
        return self.gen(x)
