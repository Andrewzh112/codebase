from torch import nn
from torch.nn import functional as F
import torchvision


class MoCo(nn.Module):
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
        fc = [nn.Linear(self.encoder.fc.in_features, args.feature_dim)]
        if args.mlp:
            fc.extend([nn.ReLU(), nn.Linear(args.feature_dim, args.feature_dim)])
        self.encoder.fc = nn.Sequential(*fc)

    def forward(self, x):
        feature = self.encoder(x)
        return F.normalize(feature, dim=-1)
