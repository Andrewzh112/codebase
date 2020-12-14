import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import argparse
from tqdm import tqdm
import os
from datetime import datetime
from pathlib import Path

from dcgan.data import get_loaders
from networks.utils import load_weights
from sagan.model import Generator, Discriminator
from sagan.loss import Hinge_loss, Wasserstein_GP_Loss


parser = argparse.ArgumentParser()

# model parameters
parser.add_argument('--h_dim', type=int, default=64, help='model dimensions multiplier')
parser.add_argument('--z_dim', type=int, default=100, help='dimension of random noise latent vector')

# data paramters
parser.add_argument('--img_size', type=int, default=128, help='H, W of the input images')
parser.add_argument('--img_channels', type=int, default=3, help='Numer of channels for images')
parser.add_argument('--crop_size', type=int, default=128, help='H, W of the input images')
parser.add_argument('--img_ext', type=str, default='jpg', help='The extension of the image files')
parser.add_argument('--download', action="store_true", default=False, help='If auto download CelebA dataset')

# training parameters
parser.add_argument('--lr_G', type=float, default=0.0001, help='Learning rate for generator')
parser.add_argument('--lr_D', type=float, default=0.0004, help='Learning rate for discriminator')
parser.add_argument('--betas', type=tuple, default=(0.0, 0.9), help='Betas for Adam optimizer')
parser.add_argument('--lambda_gp', type=float, default=10., help='Gradient penalty term')
parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--continue_train', action="store_true", default=False, help='Whether to save samples locally')
parser.add_argument('--devices', type=list, default=[0, 1], help='List of training devices')
parser.add_argument('--criterion', type=str, default='hinge', help='Loss function [hinge, wasserstein-gp]')

# logging parameters
parser.add_argument('--data_path', type=str, default='data/img_align_celeba', help='Path to where image data is located')
parser.add_argument('--cpt_interval', type=int, default=500, help='Checkpoint interval')
parser.add_argument('--save_local_samples', action="store_true", default=False, help='Whether to save samples locally')
parser.add_argument('--sample_size', type=int, default=64, help='Numbers of images to log')
parser.add_argument('--checkpoint_dir', type=str, default='sagan/checkpoint', help='Path to where model weights will be saved')
parser.add_argument('--sample_dir', type=str, default='sagan/samples', help='Path to where generated samples will be saved')
parser.add_argument('--log_dir', type=str, default='sagan/logs', help='Path to where logs will be saved')
opt = parser.parse_args()


def train():
    # creating dirs if needed
    Path(opt.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(opt.log_dir).mkdir(parents=True, exist_ok=True)
    if opt.save_local_samples:
        Path(opt.sample_dir).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(opt.log_dir + f'/{int(datetime.now().timestamp()*1e6)}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = torch.nn.DataParallel(Generator(opt.h_dim, opt.z_dim, opt.img_channels, opt.img_size), device_ids=opt.devices).to(device)
    D = torch.nn.DataParallel(Discriminator(opt.img_channels, opt.h_dim, opt.img_size), device_ids=opt.devices).to(device)

    if opt.criterion == 'wasserstein-gp':
        criterion = Wasserstein_GP_Loss(opt.lambda_gp)
    elif opt.criterion == 'hinge':
        criterion = Hinge_loss()
    else:
        raise NotImplementedError('Please choose criterion [hinge, wasserstein-gp]')
    optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr_G, betas=opt.betas)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr_D, betas=opt.betas)

    loader = get_loaders(opt.data_path, opt.img_ext, opt.crop_size,
                         opt.img_size, opt.batch_size, opt.download)
    # sample fixed z to see progress through training
    fixed_z = torch.randn(opt.sample_size, opt.z_dim).to(device)

    # if continue training, load weights, otherwise starting epoch=0
    if opt.continue_train:
        start_epoch = load_weights(state_dict_path=opt.checkpoint_dir,
                                   models=[D, G],
                                   model_names=['D', 'G'],
                                   optimizers=[optimizer_D, optimizer_G],
                                   optimizer_names=['optimizer_D', 'optimizer_G'],
                                   return_val='start_epoch')
    else:
        start_epoch = 0
    pbar = tqdm(range(start_epoch, opt.n_epochs))
    ckpt_iter = 0

    for epoch in pbar:
        d_losses, g_losses = [], []
        D.train()
        G.train()
        for batch_idx, reals in enumerate(loader):
            # preping data
            reals = reals.to(device)
            z = torch.randn(reals.size(0), opt.z_dim).to(device)

            # forward generator
            optimizer_G.zero_grad()
            fakes = G(z)

            # compute loss & update gen
            g_loss = criterion(fake_logits=D(fakes), mode='generator')
            g_loss.backward()
            optimizer_G.step()

            # forward discriminator
            optimizer_D.zero_grad()
            logits_fake = D(fakes.detach())
            logits_real = D(reals)

            # compute loss & update disc
            d_loss = criterion(fake_logits=logits_fake, real_logits=logits_real, mode='discriminator')

            # if wgangp, calculate gradient penalty and add to current d_loss
            if opt.criterion == 'wasserstein-gp':
                interpolates = criterion.get_interpolates(reals, fakes)
                interpolated_logits = D(interpolates)
                grad_penalty = criterion.grad_penalty_loss(interpolates, interpolated_logits)
                d_loss = d_loss + grad_penalty

            d_loss.backward()
            optimizer_D.step()

            # logging
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            pbar.set_postfix({
                'G Loss': g_loss.item(),
                'D Loss': d_loss.item(),
                'Batch ID': batch_idx})

            # tensorboard logging samples, not logging first iteration
            if batch_idx % opt.cpt_interval == 0:
                ckpt_iter += 1
                G.eval()
                # generate image from fixed noise vector
                with torch.no_grad():
                    samples = G(fixed_z)
                    samples = (samples + 1) / 2

                # save locally
                if opt.save_local_samples:
                    torchvision.utils.save_image(samples, f'{opt.sample_dir}/Interval_{ckpt_iter}.{opt.img_ext}')

                # save sample and loss to tensorboard
                writer.add_image('Generated Images', torchvision.utils.make_grid(samples), global_step=ckpt_iter)
                writer.add_scalars("Train Losses", {
                    "Discriminator Loss": sum(d_losses) / len(d_losses),
                    "Generator Loss": sum(g_losses) / len(g_losses)
                }, global_step=ckpt_iter)

                # resetting
                G.train()

        # printing loss and save weights
        tqdm.write(
            f'Epoch {epoch + 1}/{opt.n_epochs}, \
                Discriminator loss: {sum(d_losses) / len(d_losses):.3f}, \
                Generator Loss: {sum(g_losses) / len(g_losses):.3f}'
        )
        torch.save({
                'D': D.state_dict(),
                'G': G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'start_epoch': epoch + 1
            }, f"{opt.checkpoint_dir}/SA_GAN.pth")


if __name__ == "__main__":
    train()
