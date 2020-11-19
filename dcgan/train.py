import argparse
import torch
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from data import get_loaders
from models import Generator, Discriminator


parser = argparse.ArgumentParser()
parser.add_argument('--img_channels', type=int, default=3, help='Numer of channels for images')
parser.add_argument('--h_dim', type=float, default=64, help='model dimensions multiplier')
parser.add_argument('--z_dim', type=float, default=100, help='dimension of random noise latent vector')
parser.add_argument('--img_size', type=int, default=64, help='H, W of the input images')
parser.add_argument('--crop_size', type=int, default=128, help='H, W of the input images')
parser.add_argument('--data_path', type=str, default='celebA', help='Path to where image data is located')
parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden dimensions for model')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
parser.add_argument('--betas', type=tuple, default=(0.5, 0.999), help='Betas for Adam optimizer')
parser.add_argument('--n_epochs', type=int, default=25, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--sample_size', type=int, default=28, help='Numbers of images to log')
parser.add_argument('--img_ext', type=str, default='jpg', help='The extension of the image files')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Path to where model weights will be saved')
parser.add_argument('--log_dir', type=str, default='logs', help='Path to where logs will be saved')
opt = parser.parse_args()


def train():
    loader = get_loaders(opt)
    writer = SummaryWriter(opt.log_dir + f'/{int(datetime.now().timestamp()*1e6)}')
    G = Generator(opt)
    D = Discriminator(opt)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=opt.betas)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=opt.betas)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.to(device)
    D.to(device)
    fixed_z = torch.randn(opt.sample_size, opt.z_dim).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(opt.n_epochs)):
        d_losses, g_losses = [], []
        for reals in loader:
            reals = reals.to(device)
            z = torch.randn(reals.size(0), opt.z_dim).to(device)
            fakes = G(z)
            logits_fakes = D(fakes.detach())
            d_fake_loss = criterion(logits_fakes, torch.zeros_like(logits_fakes))
            logits_real = D(reals)
            d_real_loss = criterion(logits_real, torch.ones_like(logits_fakes))
            d_loss = d_fake_loss + d_real_loss
            g_loss = criterion(D(fakes), torch.ones_like(logits_fakes))
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            d_loss.backward()
            g_loss.backward()
            optimizer_D.step()
            optimizer_G.step()
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

        with torch.no_grad():
            samples = G(fixed_z)
        samples = ((samples + 1) / 2).view(-1, opt.img_channels, opt.img_size, opt.img_size)
        writer.add_image('Generated Images', torchvision.utils.make_grid(samples), global_step=epoch)
        writer.add_scalars("Train Losses", {
            "Discriminator Loss": sum(d_losses) / len(d_losses),
            "Generator Loss": sum(g_losses) / len(g_losses)
        })
        tqdm.write(
            f'Epoch {epoch + 1}/{opt.n_epochs}, \
                Train Disc loss: {sum(d_losses) / len(d_losses):.3f}, \
                Train Gen Loss: {sum(g_losses) / len(g_losses):.3f}'
        )


if __name__ == "__main__":
    train()
