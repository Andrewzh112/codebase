from torch import nn
import torch
from torch.utils.data import DataLoader
from utils.data import SimpleDataset
from torch.nn import functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, mode=None, activation='relu', normalization='bn', kernel_size=None):
        super().__init__()
        # typical convolution configs
        if mode == 'up':
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 2, 1, bias=False)
        elif mode == 'down':
            if kernel_size is None:
                kernel_size = 4
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, 2, 1, bias=False)
        else:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1, bias=False)

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


class SA_Conv2d(nn.Module):
    """SAGAN"""
    def __init__(self, in_channels, K=8, down_sample=True):
        super().__init__()
        self.f = nn.Conv2d(in_channels, in_channels // K, kernel_size=1)
        self.g = nn.Conv2d(in_channels, in_channels // K, kernel_size=1)
        self.h = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.v = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)

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
        h = h.view(B, 4 * C // K, HW_prime)                                 # B x (C/2) x (HW)'

        beta = self._dot_product_softmax(f, g)                              # B x (HW) x (HW)'
        s = torch.einsum('ijk,ilk->ijl', h, beta).view(B, 4 * C // K, H, W) # B x (C/2) x H x W
        return self.gamma * self.v(s) + x                                   # B x C x H x W
