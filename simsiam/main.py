from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

import argparse
from tqdm import tqdm
from pathlib import Path
from warnings import simplefilter

from model import SimSiam
from data import GaussianBlur, CIFAR10Pairs

simplefilter(action='ignore', category=ConvergenceWarning)
parser = argparse.ArgumentParser(description='Train SimSiam')

# training configs
parser.add_argument('--lr', default=0.03, type=float, help='initial learning rate')
parser.add_argument('--epochs', default=800, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')

# simsiam model configs
parser.add_argument('-a', '--backbone', default='resnet18')
parser.add_argument('--hidden_dim', default=2048, type=int, help='feature dimension')
parser.add_argument('--bottleneck_dim', default=512, type=int, help='bottleneck dimension')
parser.add_argument('--num_encoder_fcs', default=2, type=int, help='number of layers of fcs for encoder')

# knn monitor
parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor')

# misc.
parser.add_argument('--data_root', default='../data', type=str, help='path to data')
parser.add_argument('--logs_root', default='logs', type=str, help='path to logs')
parser.add_argument('--check_point', default='check_point/simsiam.pth', type=str, help='path to model weights')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cosine_loss(p, z):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p @ z.T).mean()


if __name__ == '__main__':
    """https://github.com/facebookresearch/moco"""

    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])

    test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])

    train_data = CIFAR10Pairs(root=args.data_root, train=True, transform=train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=28)
    feature_data = CIFAR10(root=args.data_root, train=True, transform=test_transform, download=True)
    feature_loader = DataLoader(feature_data, batch_size=args.batch_size, shuffle=False, num_workers=28)
    test_data = CIFAR10(root=args.data_root, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=28)

    writer = SummaryWriter(args.logs_root)
    model = SimSiam(args).to(device)
    Path(args.check_point.split('/')[0]).mkdir(parents=True, exist_ok=True)
    Path(args.logs_root).mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(args.epochs * 0.005))

    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        model.train()
        train_losses = []
        for x1, x2 in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2, p1, p2 = model(x1, x2)
            # symmetric loss
            loss = (cosine_loss(p1, z2) + cosine_loss(p2, z1)) / 2
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': round(loss.item(), 3)})
        writer.add_scalar('Train Loss', sum(train_losses) / len(train_losses), global_step=epoch)

        model.eval()
        feature_bank, targets = [], []
        # get current featuremaps & fit LR
        for data, target in feature_loader:
            data = data.to(device)
            with torch.no_grad():
                feature = model(data, istrain=False)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            targets.append(target)
        feature_bank = torch.cat(feature_bank, dim=0).cpu().numpy()
        feature_labels = torch.cat(targets, dim=0).numpy()

        linear_classifier = LogisticRegression()
        linear_classifier.fit(feature_bank, feature_labels)

        y_preds, y_trues = [], []
        for data, target in test_loader:
            data = data.to(device)
            with torch.no_grad():
                feature = model(data, istrain=False)
            feature = F.normalize(feature, dim=1).cpu().numpy()
            y_preds.extend(linear_classifier.predict(feature).tolist())
            y_trues.append(target)
        y_trues = torch.cat(y_trues, dim=0).numpy()
        top1acc = accuracy_score(y_trues, y_preds) * 100
        writer.add_scalar('Top Acc @ 1', top1acc, global_step=epoch)

        tqdm.write('#########################################################')
        tqdm.write(
            f'Epoch {epoch + 1}/{args.epochs}, \
                Train Loss: {sum(train_losses) / len(train_losses):.3f}, \
                Top Acc @ 1: {top1acc:.3f}, \
                Learning Rate: {scheduler.get_last_lr()}'
        )
        torch.save(model.state_dict(), args.check_point)
        scheduler.step()
