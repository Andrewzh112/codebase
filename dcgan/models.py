from torch import nn


class Discriminator(nn.Module):
    def __init__(self, img_channels, h_dim, img_size):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, h_dim, 4, 2, 1),
            conv_bn_relu(h_dim, h_dim*2, 4, 2, 'lrelu', 'up'),
            conv_bn_relu(h_dim*2, h_dim*4, 4, 2, 'lrelu', 'up'),
            conv_bn_relu(h_dim*4, h_dim*8, 4, 2, 'lrelu', 'up'),
            nn.Flatten(),
            nn.Linear(h_dim*8 * (img_size // (2 ** 4)) ** 2, 1)
        )
        initialize_weights(self)

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
            conv_bn_relu(h_dim*8, h_dim*4, 4, 2, 'relu', 'down'),
            conv_bn_relu(h_dim*4, h_dim*2, 4, 2, 'relu', 'down'),
            conv_bn_relu(h_dim*2, h_dim, 4, 2, 'relu', 'down'),
            nn.ConvTranspose2d(h_dim, img_channels, 4, 2, 1),
            nn.Sigmoid()
        )
        initialize_weights(self)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.project(x)
        x = x.view(batch_size, self.h_dim*8, self.s16, self.s16)
        return self.gen(x)


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, activation, mode):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, bias=False) \
            if mode == 'up' else \
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels, momentum=0.9),
        nn.LeakyReLU(0.2) if activation == 'lrelu' else nn.ReLU())


def initialize_weights(model, nonlinearity='leaky_relu'):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                m.weight,
                mode='fan_out',
                nonlinearity=nonlinearity
            )
        elif isinstance(m, (nn.Linear, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)
