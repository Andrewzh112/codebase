from torch import nn


def initialize_modules(model, nonlinearity='leaky_relu'):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                m.weight,
                mode='fan_out',
                nonlinearity=nonlinearity
            )
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)
