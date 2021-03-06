from torch import nn
from torch.nn import functional as F
import torchvision
from networks.layers import ConvNormAct


class MoCo(nn.Module):
    def __init__(self, feature_dim, backbone, mlp):
        super().__init__()
        if backbone == 'resnet18':
            self.encoder = torchvision.models.resnet18(progress=False)
        elif backbone == 'resnet34':
            self.encoder = torchvision.models.resnet34(progress=False)
        elif backbone == 'resnet50':
            self.encoder = torchvision.models.resnet50(progress=False)
        elif backbone == 'resnet101':
            self.encoder = torchvision.models.resnet101(progress=False)
        elif backbone == 'resnet152':
            self.encoder = torchvision.models.resnet152(progress=False)
        elif backbone == 'simple':
            self.encoder = nn.Sequential(
                ConvNormAct(3, 32, mode='down'),
                ConvNormAct(32, 64, mode='down'),
                ConvNormAct(64, 128, mode='down'),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            raise NotImplementedError

        # disabling last layers, also saving the feature dimension for inference
        if backbone != 'simple':
            self.out_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:    
            self.out_features = 128

        # projector after feature extractor
        fc = [nn.Linear(self.out_features, feature_dim)]
        if mlp:
            fc.extend([nn.ReLU(), nn.Linear(feature_dim, feature_dim)])
        self.projector = nn.Sequential(*fc)

    def forward(self, x):
        feature = self.encoder(x)

        # only project when training, output features otherwise
        if self.training:
            feature = self.projector(feature)
        return F.normalize(feature, dim=1)
