import argparse
import torch
import torchvision
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
from dcgan.data import get_loaders
from sagan.model import Generator, Discriminator
from sagan.loss import SAGAN_Hinge_loss


parser = argparse.ArgumentParser()

# model parameters
parser.add_argument('--h_dim', type=float, default=64, help='model dimensions multiplier')
parser.add_argument('--z_dim', type=float, default=100, help='dimension of random noise latent vector')
parser.add_argument('--n_res_blocks', type=int, default=9, help='Number of ResNet Blocks')

# data paramters
parser.add_argument('--img_size', type=int, default=128, help='H, W of the input images')
parser.add_argument('--img_channels', type=int, default=3, help='Numer of channels for images')
parser.add_argument('--crop_size', type=int, default=128, help='H, W of the input images')
parser.add_argument('--img_ext', type=str, default='jpg', help='The extension of the image files')

# training parameters
parser.add_argument('--lr_G', type=float, default=0.0001, help='Learning rate for generator')
parser.add_argument('--lr_D', type=float, default=0.0004, help='Learning rate for discriminator')
parser.add_argument('--betas', type=tuple, default=(0.0, 0.99), help='Betas for Adam optimizer')
parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

# logging parameters
parser.add_argument('--data_path', type=str, default='data/img_align_celeba', help='Path to where image data is located')
parser.add_argument('--save_every', type=int, default=10, help='Save interval')
parser.add_argument('--save_local_samples', action="store_true", default=False, help='Whether to save samples locally')
parser.add_argument('--sample_size', type=int, default=64, help='Numbers of images to log')
parser.add_argument('--checkpoint_dir', type=str, default='sagan/checkpoint', help='Path to where model weights will be saved')
parser.add_argument('--sample_dir', type=str, default='sagan/samples', help='Path to where generated samples will be saved')
parser.add_argument('--log_dir', type=str, default='sagan/logs', help='Path to where logs will be saved')
opt = parser.parse_args()


def train():
    writer = SummaryWriter(opt.log_dir + f'/{int(datetime.now().timestamp()*1e6)}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(opt.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(opt.logs_root).mkdir(parents=True, exist_ok=True)
    Path(opt.sample_dir).mkdir(parents=True, exist_ok=True)

    G = Generator(opt)
    D = Discriminator(opt)
    G.to(device)
    D.to(device)

    criterion = SAGAN_Hinge_loss()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr_G, betas=opt.betas)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr_D, betas=opt.betas)

    loader = get_loaders(opt)
    fixed_z = torch.randn(opt.sample_size, opt.z_dim).to(device)

    for epoch in tqdm(range(opt.n_epochs)):
        d_losses, g_losses = [], []
        D.train()
        G.train()
        for reals in loader:
            reals = reals.to(device)
            z = torch.randn(reals.size(0), opt.z_dim).to(device)
            fakes = G(z)
            logits_fake = D(fakes.detach())
            logits_real = D(reals)
            d_loss = criterion(fake_logits=logits_fake, real_logits=logits_real, mode='discriminator')
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            g_loss = criterion(fake_logits=D(fakes), mode='generator')
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        G.eval()
        # generate image from fixed noise vector
        with torch.no_grad():
            samples = G(fixed_z)
        samples = ((samples + 1) / 2).view(-1, opt.img_channels, opt.img_size, opt.img_size)

        # logging to tensorboard
        writer.add_image('Generated Images', torchvision.utils.make_grid(samples), global_step=epoch)
        writer.add_scalars("Train Losses", {
            "Discriminator Loss": sum(d_losses) / len(d_losses),
            "Generator Loss": sum(g_losses) / len(g_losses)
        }, global_step=epoch)

        if opt.save_local_samples:
            torchvision.utils.save_image(samples, f'{opt.sample_dir}/Epoch_{epoch}.png')

        # printing loss and save weights
        tqdm.write(
            f'Epoch {epoch + 1}/{opt.n_epochs}, \
                Train Disc loss: {sum(d_losses) / len(d_losses):.3f}, \
                Train Gen Loss: {sum(g_losses) / len(g_losses):.3f}'
        )
        if (epoch + 1) % opt.save_every == 0:
            torch.save({
                    'D': D.state_dict(),
                    'G': G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'optimizer_G': optimizer_G.state_dict()
                }, f"{opt.checkpoint_dir}/cycleGAN_{epoch + 1}.pth")


if __name__ == "__main__":
    train()
