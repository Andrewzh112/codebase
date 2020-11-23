from torch import nn


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


def initialize_weights(model, nonlinearity='leaky_relu'):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                m.weight,
                mode='fan_out',
                nonlinearity=nonlinearity
            )
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)


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
                stride=2,
                padding_mode='reflect'
            ),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim * 4,
                kernel_size=3,
                padding=1,
                stride=2,
                padding_mode='reflect'
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
        initialize_weights(self, nonlinearity='relu')

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super().__init__()
        self.disc = nn.Sequential(
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
                    kernel_size=4,
                    padding=1,
                    stride=2,
                    padding_mode='reflect'
                ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                    hidden_dim * 2,
                    hidden_dim * 4,
                    kernel_size=4,
                    padding=1,
                    stride=2,
                    padding_mode='reflect'
                ),
            nn.InstanceNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                    hidden_dim * 4,
                    hidden_dim * 8,
                    kernel_size=4,
                    padding=1,
                    stride=2,
                    padding_mode='reflect'
                ),
            nn.InstanceNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                hidden_dim * 8,
                out_channels=1,
                kernel_size=1
            )
        )
        initialize_weights(self)

    def forward(self, x):
        return self.disc(x) 
