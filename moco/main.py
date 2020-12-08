from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from moco.model import MoCo
from moco.utils import (GaussianBlur, CIFAR10Pairs, MoCoLoss,
                        MemoryBank, momentum_update, get_momentum_encoder)
from networks.layers import Linear_Probe
from networks.utils import load_weights
from utils.contrastive import get_feature_label

parser = argparse.ArgumentParser(description='Train MoCo')

# training configs
parser.add_argument('--lr', default=0.03, type=float, help='initial learning rate')
parser.add_argument('--continue_train', action="store_true", default=False, help='continue training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=0.0001, type=float, metavar='W', help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
parser.add_argument('--temperature', default=0.07, type=float, help='temperature for loss fn')
parser.add_argument('--K', default=4096, type=int, help='memory bank size')
parser.add_argument('--m', default=0.999, type=float, help='momentum update parameter')

# moco model configs
parser.add_argument('-a', '--backbone', default='resnet18')
parser.add_argument('--feature_dim', default=128, type=int, help='feature dimension')
parser.add_argument('--mlp', default=True, type=bool, help='feature dimension')

# misc.
parser.add_argument('--data_root', default='data', type=str, help='path to data')
parser.add_argument('--logs_root', default='moco/logs', type=str, help='path to logs')
parser.add_argument('--check_point', default='moco/check_point/moco.pth', type=str, help='path to model weights')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
    momentum_data = CIFAR10(root=args.data_root, train=True, transform=test_transform, download=True)
    momentum_loader = DataLoader(momentum_data, batch_size=args.batch_size, shuffle=False, num_workers=28)
    test_data = CIFAR10(root=args.data_root, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=28)

    Path('/'.join(args.check_point.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    Path(args.logs_root).mkdir(parents=True, exist_ok=True)

    f_q = torch.nn.DataParallel(MoCo(args), device_ids=[0, 1]).to(device)
    f_k = get_momentum_encoder(f_q)

    criterion = MoCoLoss(args.temperature)
    optimizer = torch.optim.SGD(f_q.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,
                                                          lambda epoch: 0.1 if epoch in (120, 160) else 1)
    memo_bank = MemoryBank(f_k, device, momentum_loader, args.K)
    writer = SummaryWriter(args.logs_root + f'/{int(datetime.now().timestamp()*1e6)}')

    start_epoch = 0
    if args.continue_train:
        start_epoch = load_weights(state_dict_path=args.check_point,
                                   models=[f_q, f_k],
                                   model_names=['f_q', 'f_k'],
                                   optimizers=[optimizer],
                                   optimizer_names=['optimizer'],
                                   return_val='start_epoch')

    pbar = tqdm(range(start_epoch, args.epochs))
    for epoch in pbar:
        train_losses = []
        for x1, x2 in train_loader:
            q1, q2 = f_q(x1), f_q(x2)
            with torch.no_grad():
                k1, k2 = f_k(x1), f_k(x2)
            loss = criterion(q1, k2, memo_bank) + criterion(q2, k1, memo_bank)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            k = torch.cat([k1, k2], dim=0)
            memo_bank.dequeue_and_enqueue(k)
            with torch.no_grad():
                momentum_update(f_k, f_q, args.m)
            train_losses.append(loss.item())
            pbar.set_postfix({'Loss': loss.item(), 'Learning Rate': scheduler.get_last_lr()[0]})

        writer.add_scalar('Train Loss', sum(train_losses) / len(train_losses), global_step=epoch)
        scheduler.step()

        f_q.eval()
        # extract features as training data
        feature_bank, feature_labels = get_feature_label(f_q, momentum_loader, device, normalize=True)

        linear_classifier = Linear_Probe(len(momentum_data.classes), hidden_dim=args.feature_dim).to(device)
        linear_classifier.fit(feature_bank, feature_labels)

        # using linear classifier to predict test data
        y_preds, y_trues = get_feature_label(f_q, test_loader, device, normalize=True, predictor=linear_classifier)
        top1acc = y_trues.eq(y_preds).sum().item() / y_preds.size(0)

        writer.add_scalar('Top Acc @ 1', top1acc, global_step=epoch)
        writer.add_scalar('Representation Standard Deviation', feature_bank.std(), global_step=epoch)
        tqdm.write(f'Epoch: {epoch + 1}/{args.epochs}, \
                Training Loss: {sum(train_losses) / len(train_losses)}, \
                Top Accuracy @ 1: {top1acc}, \
                Representation STD: {feature_bank.std()}')
        torch.save({
            'f_q': f_q.state_dict(),
            'f_k': f_k.state_dict(),
            'optimizer': optimizer.state_dict(),
            'start_epoch': epoch + 1},
            args.check_point)
