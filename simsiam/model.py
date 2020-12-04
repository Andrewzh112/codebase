"""https://github.com/facebookresearch/moco"""

import torch
from torch import nn
from torch.nn import functional as F
import torchvision


class SimSiam(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone == 'resnet18':
            self.encoder = torchvision.models.resnet18(progress=False)
        elif args.backbone == 'resnet34':
            self.encoder = torchvision.models.resnet34(progress=False)
        elif args.backbone == 'resnet50':
            self.encoder = torchvision.models.resnet50(progress=False)
        elif args.backbone == 'resnet101':
            self.encoder = torchvision.models.resnet101(progress=False)
        elif args.backbone == 'resnet152':
            self.encoder = torchvision.models.resnet152(progress=False)
        else:
            raise NotImplementedError

        fc = []
        for i in range(args.num_encoder_fcs):
            # dim of resnet features
            if i == 0:
                in_features = self.encoder.fc.in_features
            else:
                in_features = args.hidden_dim
            fc.append(nn.Linear(in_features, args.hidden_dim, bias=False))
            fc.append(nn.BatchNorm1d(args.hidden_dim))
            # no relu for output layer
            if i < args.num_encoder_fcs - 1:
                fc.append(nn.ReLU())
        self.encoder.fc = nn.Sequential(*fc)

        self.projector = nn.Sequential(
            nn.Linear(args.hidden_dim, args.bottleneck_dim, bias=False),
            nn.BatchNorm1d(args.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(args.bottleneck_dim, args.hidden_dim),
        )

    def forward(self, x1, x2=None, istrain=True):
        if istrain:
            z1, z2 = self.encoder(x1), self.encoder(x2)
            p1, p2 = self.projector(z1), self.projector(z2)
            return z1, z2, p1, p2
        else:
            return self.encoder(x1)

    def cosine_loss(self, p, z):
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1).detach()
        return -torch.einsum('ij,ij->i', p, z).mean()
