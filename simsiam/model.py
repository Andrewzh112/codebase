"""https://github.com/facebookresearch/moco"""

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.utils.data import DataLoader
from simsiam.data import SimpleDataset


class Linear_Classifier(nn.Module):
    def __init__(self, args, num_classes, epochs=2000, lr=1e-3):
        super().__init__()
        self.fc = nn.Linear(args.hidden_dim, num_classes)
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.fc(x)

    def loss(self, y_hat, y):
        return self.criterion(y_hat, y)

    def fit(self, x, y):
        dataset = SimpleDataset(x, y)
        loader = DataLoader(dataset, batch_size=2056, shuffle=True)
        self.train()
        for _ in range(self.epochs):
            for features, labels in loader:
                y_hat = self.forward(features)
                loss = self.loss(y_hat, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return torch.argmax(predictions, dim=1)


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
