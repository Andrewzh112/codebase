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
        elif normalization == 'gn':
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
