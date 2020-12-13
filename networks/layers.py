from torch import nn
import torch
from torch.utils.data import DataLoader
from utils.data import SimpleDataset
from torch.nn import functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type='basic', mode=None, activation='relu', normalization='bn', kernel_size=None):
        super().__init__()
        # type of convolution
        if conv_type == 'basic' and mode is None or mode == 'down':
            conv = nn.Conv2d
        elif conv_type == 'sn' and mode is None or mode == 'down':
            conv = SN_Conv2d
        elif conv_type == 'basic' and mode is None or mode == 'up':
            conv = nn.ConvTranspose2d
        elif conv_type == 'sn' and mode is None or mode == 'up':
            conv = SN_ConvTranspose2d
        else:
            raise NotImplementedError('Please only choose conv [basic, sn] and mode [None, down, up]')

        if mode == 'up':
            if kernel_size is None:
                kernel_size = 4
            conv = conv(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=2, padding=1, bias=False)
        elif mode == 'down':
            if kernel_size is None:
                kernel_size = 4
            conv = conv(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=2, padding=1, bias=False)
        else:
            if kernel_size is None:
                kernel_size = 3
            conv = conv(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=1, padding=1, bias=False)

        # normalization
        # TODO GroupNorm
        if normalization == 'bn':
            norm = nn.BatchNorm2d(out_channels)
        elif normalization == 'ln':
            norm = nn.LayerNorm(out_channels)
        elif normalization == 'in':
            norm = nn.InstanceNorm2d(out_channels)
        else:
            raise NotImplementedError('Please only choose normalization [bn, ln, in]')

        # activations
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'lrelu':
            act = nn.LeakyReLU(0.2)
        else:
            raise NotImplementedError('Please only choose activation [relu, lrelu]')

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
            raise NotImplementedError('Please only choose normalization [bn, ln, in]')

        # activations
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'lrelu':
            act = nn.LeakyReLU(0.2)
        else:
            raise NotImplementedError('Please only choose activation [relu, lrelu]')

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


class Linear_Probe(nn.Module):
    def __init__(self, num_classes, in_features=256, lr=30):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=0)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            self.optimizer,
            lr_lambda=lambda lr: 0.995)

    def forward(self, x):
        return self.fc(x)

    def loss(self, y_hat, y):
        return self.criterion(y_hat, y)

    def fit(self, x, y, epochs=100):
        dataset = SimpleDataset(x, y)
        loader = DataLoader(dataset, batch_size=2056, shuffle=True)
        self.train()
        for _ in range(epochs):
            for features, labels in loader:
                y_hat = self.forward(features)
                loss = self.loss(y_hat, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return torch.argmax(predictions, dim=1)


class SN_Conv2d(nn.Module):
    def __init__(self, eps=1e-12, **kwargs):
        super().__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps)

    def forward(self, x):
        return self.conv(x)


class SN_ConvTranspose2d(nn.Module):
    def __init__(self, eps=1e-12, **kwargs):
        super().__init__()
        self.conv = nn.utils.spectral_norm(nn.ConvTranspose2d(**kwargs), self.eps)

    def forward(self, x):
        return self.conv(x)


class SN_Linear(nn.Module):
    def __init__(self, eps=1e-12, **kwargs):
        super().__init__()
        self.fc = nn.utils.spectral_norm(nn.Linear(**kwargs), eps)

    def forward(self, x):
        return self.fc(x)


class SN_Embedding(nn.Module):
    def __init__(self, eps=1e-12, **kwargs):
        super().__init__()
        self.embed = nn.utils.spectral_norm(nn.Embedding(**kwargs), eps)

    def forward(self, x):
        return self.Embedding(x)


class SA_Conv2d(nn.Module):
    """SAGAN"""
    def __init__(self, in_channels, conv=SN_Conv2d, K=8, down_sample=True):
        super().__init__()

        self.f = conv(in_channels=in_channels, out_channels=in_channels // K, kernel_size=1)
        self.g = conv(in_channels=in_channels, out_channels=in_channels // K, kernel_size=1)
        self.h = conv(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1)
        self.v = conv(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1)

        # adaptive attention weight
        self.gamma = nn.Parameter(torch.tensor(0., requires_grad=True))

        self.down_sample = down_sample
        self.K = K

    def _dot_product_softmax(self, f, g):
        s = torch.einsum('ijk,ijl->ikl', f, g)
        beta = torch.softmax(s, dim=-1)
        return beta

    def forward(self, x):
        """
        b - batch size
        c - channels
        h - height
        w - width
        k - shrinking factor
        """
        # tracking shapes
        B, C, H, W = x.size()
        K = self.K
        HW_prime = H * W

        # get qkv's
        f = self.f(x).view(B, C // K, H * W)                                # B x (C/K) x (HW)
        g = self.g(x)                                                       # B x (C/K) x H x W
        h = self.h(x)                                                       # B x (C/2) x H x W
        if self.down_sample:
            g = F.max_pool2d(g, [2, 2])                                     # B x (C/K) x (H/2) x (W/2)
            h = F.max_pool2d(h, [2, 2])                                     # B x (C/2) x (H/2) x (W/2)
            HW_prime = HW_prime // 4                                        # update (HW)'<-(HW) // 4

        g = g.view(B, C // K, HW_prime)                                     # B x (C/K) x (HW)'
        h = h.view(B, C // 2, HW_prime)                                     # B x (C/2) x (HW)'

        beta = self._dot_product_softmax(f, g)                              # B x (HW) x (HW)'
        s = torch.einsum('ijk,ilk->ijl', h, beta).view(B, C // 2, H, W)     # B x (C/2) x H x W
        return self.gamma * self.v(s) + x                                   # B x C x H x W
