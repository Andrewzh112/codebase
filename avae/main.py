import argparse
import os
import torch

from avae.model import AVAE_Trainer
from data.unlabelled import get_celeba_loaders


parser = argparse.ArgumentParser()

# image settings
parser.add_argument('--img_channels', type=int, default=3, help='Numer of channels for images')
parser.add_argument('--img_size', type=int, default=64, help='H, W of the input images')
parser.add_argument('--crop_size', type=int, default=128, help='H, W of the input images')
parser.add_argument('--download', action="store_true", default=False, help='If auto download CelebA dataset')

# model params
parser.add_argument('--z_dim', type=float, default=100, help='dimension of random noise latent vector z')
parser.add_argument('--xi_dim', type=float, default=100, help='dimension of random noise latent vector xi')
parser.add_argument('--model_dim', type=float, default=128, help='model dimensions multiplier')

# loss fn
parser.add_argument('--beta', type=float, default=5., help='Beta hyperparam for KLD Loss')

# training hyperparams
parser.add_argument('--device_ids', type=list, default=[0, 1], help='List of GPU devices')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--betas', type=tuple, default=(0.5, 0.999), help='Betas for Adam optimizer')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--data_parallel', action="store_true", default=False, help='train with data parallel')

# logging
parser.add_argument('--log_dir', type=str, default='avae/logs', help='Path to where log files will be saved')
parser.add_argument('--data_path', type=str, default='data/img_align_celeba', help='Path to where image data is located')
parser.add_argument('--sample_path', type=str, default='avae/samples', help='Path to where samples are saved')
parser.add_argument('--img_ext', type=str, default='.jpg', help='Image extentions')
parser.add_argument('--checkpoint_dir', type=str, default='avae/model_weights', help='Path to where model weights will be saved')

# for sampling
parser.add_argument('--sample_size', type=int, default=64, help='Size of sampled images')
parser.add_argument('--sample', action="store_true", default=False, help='Sample from VAE')
parser.add_argument('--walk', action="store_true", default=False, help='Walk through a feature & sample')

args = parser.parse_args()


if __name__ == '__main__':
    loader = get_celeba_loaders(args.data_path, args.img_ext, args.crop_size,
                                args.img_size, args.batch_size, args.download)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = AVAE_Trainer(args.img_channels, args.model_dim, args.img_size, args.z_dim,
                           args.xi_dim, args.lr, args.betas, args.beta, args.n_epochs,
                           args.sample_size, args.log_dir, args.checkpoint_dir, device).to(device)
    if args.data_parallel:
        trainer = torch.nn.DataParallel(trainer, device_ids=args.device_ids)
        trainer.module.fit(loader)
    else:
        trainer.fit(loader)
