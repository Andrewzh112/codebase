from torch import nn
from networks.layers import ConvNormAct, SA_Conv2d
from networks.utils import initialize_modules


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(args.img_channels, args.h_dim, 4, 2, 1),
            ConvNormAct(args.h_dim, args.h_dim*2, 'down', activation='lrelu', normalization='bn'),
            ConvNormAct(args.h_dim*2, args.h_dim*4, 'down', activation='lrelu', normalization='bn'),
            ConvNormAct(args.h_dim*4, args.h_dim*8, 'down', activation='lrelu', normalization='bn'),
            nn.Flatten(),
            nn.Linear(args.h_dim*8 * (args.img_size // (2 ** 4)) ** 2, 1)
        )
        initialize_modules(self)

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.s16 = args.img_size // 16
        self.project = nn.Linear(args.z_dim, args.h_dim*8*self.s16*self.s16)
        self.gen = nn.Sequential(
            nn.BatchNorm2d(args.h_dim*8, momentum=0.9),
            nn.ReLU(),
            ConvNormAct(args.h_dim*8, args.h_dim*4, 'up', activation='relu', normalization='bn'),
            ConvNormAct(args.h_dim*4, args.h_dim*2, 'up', activation='relu', normalization='bn'),
            SA_Conv2d(args.h_dim*2),
            ConvNormAct(args.h_dim*2, args.h_dim, 'up', activation='relu', normalization='bn'),
            nn.ConvTranspose2d(args.h_dim, args.img_channels, 4, 2, 1),
            nn.Sigmoid()
        )
        initialize_modules(self)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.project(x)
        x = x.view(batch_size, self.args.h_dim*8, self.s16, self.s16)
        return self.gen(x)
