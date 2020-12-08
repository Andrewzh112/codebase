from torch import nn
from torch.nn import init


class ResBlock(nn.Module):
    def __init__(self, in_channels, activation=None, norm=None):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        if norm is None:
            norm = nn.InstanceNorm2d(in_channels)
        self.resblock = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                padding_mode='reflect'
            ),
            norm,
            activation,
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                padding_mode='reflect'
            ),
            norm
        )

    def forward(self, x):
        return self.resblock(x)


def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def init_weights(model, init_type='normal', init_gain=0.02):
    """https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    model.apply(init_func)


class Generator(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        hidden_dim,
        n_res_blocks,
        activation=None,
        norm=None):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_dim,
                kernel_size=7,
                padding=3,
                padding_mode='reflect'
            ),
            nn.Conv2d(
                hidden_dim,
                hidden_dim * 2,
                kernel_size=3,
                padding=1,
                stride=2
            ),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim * 4,
                kernel_size=3,
                padding=1,
                stride=2
            ),
            nn.InstanceNorm2d(hidden_dim * 4),
            nn.ReLU(),
            *[ResBlock(hidden_dim * 4, activation, norm) for _ in range(n_res_blocks)],
            nn.ConvTranspose2d(
                hidden_dim * 4,
                hidden_dim * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim * 2,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim,
                output_channels,
                kernel_size=7,
                padding=3,
                padding_mode='reflect'
            ),
            nn.Tanh()
        )
        init_weights(self)

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(
                    input_channels,
                    hidden_dim,
                    kernel_size=4,
                    padding=1,
                    stride=2
                ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                    hidden_dim,
                    hidden_dim * 2,
                    kernel_size=4,
                    padding=1,
                    stride=2
                ),
            nn.InstanceNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                    hidden_dim * 2,
                    hidden_dim * 4,
                    kernel_size=4,
                    padding=1,
                    stride=2
                ),
            nn.InstanceNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                    hidden_dim * 4,
                    hidden_dim * 8,
                    kernel_size=4,
                    padding=1,
                    stride=1,
                ),
            nn.InstanceNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                hidden_dim * 8,
                out_channels=1,
                kernel_size=4,
                padding=1
            )
        )
        init_weights(self)

    def forward(self, x):
        return self.disc(x) 
