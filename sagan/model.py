from torch import nn
from networks.layers import ConvNormAct, SA_Conv2d
from networks.utils import initialize_modules


class Discriminator(nn.Module):
    def __init__(self, img_channels, h_dim, img_size):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, h_dim, 4, 2, 1),
            ConvNormAct(h_dim, h_dim*2, 'down', activation='lrelu', normalization='bn'),
            ConvNormAct(h_dim*2, h_dim*4, 'down', activation='lrelu', normalization='bn'),
            ConvNormAct(h_dim*4, h_dim*8, 'down', activation='lrelu', normalization='bn'),
            nn.Flatten(),
            nn.Linear(h_dim*8 * (img_size // (2 ** 4)) ** 2, 1)
        )
        initialize_modules(self)

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, h_dim, z_dim, img_channels, img_size):
        super().__init__()
        self.s16 = img_size // 16
        self.h_dim = h_dim
        self.project = nn.Linear(z_dim, h_dim*8*self.s16*self.s16)
        self.gen = nn.Sequential(
            nn.BatchNorm2d(h_dim*8, momentum=0.9),
            nn.ReLU(),
            ConvNormAct(h_dim*8, h_dim*4, 'up', activation='relu', normalization='bn'),
            ConvNormAct(h_dim*4, h_dim*2, 'up', activation='relu', normalization='bn'),
            SA_Conv2d(h_dim*2),
            ConvNormAct(h_dim*2, h_dim, 'up', activation='relu', normalization='bn'),
            nn.ConvTranspose2d(h_dim, img_channels, 4, 2, 1),
            nn.Sigmoid()
        )
        initialize_modules(self)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.project(x)
        x = x.view(batch_size, self.h_dim*8, self.s16, self.s16)
        return self.gen(x)
