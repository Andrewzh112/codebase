from torch import nn


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, mode=None, activation='relu', normalization='bn'):
        super().__init__()
        # typical convolution configs 
        if mode == 'up':
            conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        elif mode == 'down':
            conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        else:
            conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)

        # normalization
        # TODO GroupNorm
        if normalization == 'bn':
            norm = nn.BatchNorm2d(out_channels)
        elif normalization == 'ln':
            norm = nn.LayerNorm(out_channels)
        elif normalization == 'in':
            norm = nn.InstanceNorm2d(out_channels)
        else:
            raise NotImplementedError

        # activations
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'lrelu':
            act = nn.LeakyReLU(0.2)
        else:
            raise NotImplementedError

        self.block = nn.Sequential(
            conv,
            norm,
            act
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, activation, normalization):
        super().__init__()
        if normalization == 'bn':
            norm = nn.BatchNorm2d(in_channels)
        elif normalization == 'ln':
            norm = nn.LayerNorm(in_channels)
        elif normalization == 'in':
            norm = nn.InstanceNorm2d(in_channels)
        else:
            raise NotImplementedError

        # activations
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'lrelu':
            act = nn.LeakyReLU(0.2)
        else:
            raise NotImplementedError

        self.resblock = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
            ),
            norm,
            act,
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
            ),
            norm
        )

    def forward(self, x):
        return self.resblock(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
