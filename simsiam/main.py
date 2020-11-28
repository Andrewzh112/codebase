from torchvision import transforms
from torchvision.datasets import CIFAR10
from data import GaussianBlur, CIFAR10Pairs
from torch.utils.data import DataLoader
import argparse
import pytorch_lightning as pl
from model import SimSiam 
from pytorch_lightning.loggers import TensorBoardLogger


parser = argparse.ArgumentParser(description='Train SimSiam')

# training configs
parser.add_argument('--lr', default=0.03, type=float, help='initial learning rate')
parser.add_argument('--epochs', default=800, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='mini-batch size')
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

args = parser.parse_args()

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

    train_data = CIFAR10Pairs(root='../data', train=True, transform=train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                  drop_last=True)
    feature_data = CIFAR10(root='data', train=True, transform=test_transform, download=True)
    feature_loader = DataLoader(feature_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    test_data = CIFAR10(root='data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    logger = TensorBoardLogger("tb_logs", name="simsiam")
    model = SimSiam(args)

    trainer = pl.Trainer(logger=logger)
    trainer.fit(model, train_loader, [feature_loader, test_loader])
