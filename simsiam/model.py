"""https://github.com/facebookresearch/moco"""

from torch import nn
import torchvision
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import math


class SimSiam(pl.LightningModule):
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

        self.feature_bank = []
        self.total_num, self.total_top1 = 0, 0
        self.is_feature_data = True
        self.epoch = 0

    def forward(self, x1, x2=None, istrain=True):
        if istrain:
            z1, z2 = self.encoder(x1), self.encoder(x2)
            p1, p2 = self.projector(z1), self.projector(z2)
            return z1, z2, p1, p2
        else:
            return self.encoder(x1)

    def training_step(self, train_batch, batch_idx):
        x1, x2 = train_batch
        z1, z2, p1, p2 = self.forward(x1, x2)
        loss = (self._cosineloss(p1, z2) + self._cosineloss(p2, z1)) / 2
        self.log('train_loss', loss)
        return loss

    def train_epoch_end(self, train_losses):
        self.logger.experiment.add_scalar(
                'train loss',
                sum(train_losses) / len(train_losses),
                global_step=self.epoch)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        data, target = batch
        feature = self.forward(data, istrain=False)
        feature = F.normalize(feature, dim=1)

        if dataloader_idx == 0:
            self.feature_bank.append(feature)
            return
        else:
            if batch_idx == 0:
                # complete featurebank & setup
                self.feature_bank = torch.cat(self.feature_bank, dim=0).t().contiguous()
                self.feature_labels = torch.tensor(self.args.targets, device=self.feature_bank.device)
                self.is_feature_data = False
            # pred_labels = [self.knn_predict(f, self.args.knn_k, self.args.knn_t) for f in feature]
            pred_labels = self.knn_predict(feature, self.args.knn_k, self.args.knn_t)
            self.total_num += data.size(0)
            self.total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            return self.total_top1 / self.total_num * 100

    def validation_epoch_end(self, top1_acc):
        if not self.is_feature_data:
            # resetting & logging
            top1_acc = top1_acc[1]
            self.feature_bank = []
            self.total_num, self.total_top1 = 0, 0
            self.log('top 1 accuracy', sum(top1_acc) / len(top1_acc))
            self.logger.experiment.add_scalar(
                'top 1 accuracy',
                sum(top1_acc) / len(top1_acc),
                global_step=self.epoch)
            self.is_feature_data = True
            self.epoch += 1

    def knn_predict(self, feature, knn_k, knn_t):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, self.feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # free memory
        sim_matrix = None
        # [B, K]
        sim_labels = torch.gather(self.feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, self.args.classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter_(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, self.args.classes) * sim_weight.unsqueeze(-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr,
                                    momentum=self.args.momentum, weight_decay=self.args.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs)
        return [optimizer], [scheduler]

    def _cosineloss(self, p, z):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p @ z.T).mean()
