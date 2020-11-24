from torch import nn


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(args.img_channels, args.h_dim, 4, 2, 1),
            conv_bn_relu(args.h_dim, args.h_dim*2, 4, 2, 'lrelu', 'up'),
            conv_bn_relu(args.h_dim*2, args.h_dim*4, 4, 2, 'lrelu', 'up'),
            conv_bn_relu(args.h_dim*4, args.h_dim*8, 4, 2, 'lrelu', 'up'),
            nn.Flatten(),
            nn.Linear(args.h_dim*8*4*4, 1)
        )
        initialize_weights(self)

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
            conv_bn_relu(args.h_dim*8, args.h_dim*4, 4, 2, 'relu', 'down'),
            conv_bn_relu(args.h_dim*4, args.h_dim*2, 4, 2, 'relu', 'down'),
            conv_bn_relu(args.h_dim*2, args.h_dim, 4, 2, 'relu', 'down'),
            nn.ConvTranspose2d(args.h_dim, args.img_channels, 4, 2, 1),
            nn.Tanh()
        )
        initialize_weights(self)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.project(x)
        x = x.view(batch_size, self.args.h_dim*8, self.s16, self.s16)
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
