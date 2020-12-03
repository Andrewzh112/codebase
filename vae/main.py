import torch
import torchvision
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from vae.data import get_loaders
from vae.model import VAE
from vae.loss import VAELoss


parser = argparse.ArgumentParser()
parser.add_argument('--img_channels', type=int, default=3, help='Numer of channels for images')
parser.add_argument('--model_dim', type=float, default=64, help='model dimensions multiplier')
parser.add_argument('--z_dim', type=float, default=100, help='dimension of random noise latent vector')
parser.add_argument('--img_size', type=int, default=64, help='H, W of the input images')
parser.add_argument('--crop_size', type=int, default=128, help='H, W of the input images')
parser.add_argument('--n_res_blocks', type=int, default=9, help='Number of ResNet Blocks for generators')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for generators')
parser.add_argument('--betas', type=tuple, default=(0.5, 0.999), help='Betas for Adam optimizer')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--sample_size', type=int, default=32, help='Size of sampled images')
parser.add_argument('--log_dir', type=str, default='vae/logs', help='Path to where log files will be saved')
parser.add_argument('--data_path', type=str, default='data/img_align_celeba', help='Path to where image data is located')
parser.add_argument('--device_ids', type=list, default=[0, 1], help='List of GPU devices')
parser.add_argument('--img_ext', type=str, default='.jpg', help='Image extentions')
parser.add_argument('--checkpoint_dir', type=str, default='vae/model_weights', help='Path to where model weights will be saved')
args = parser.parse_args()


if __name__ == '__main__':
    writer = SummaryWriter(args.log_dir)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    loader = get_loaders(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.nn.DataParallel(VAE(args), device_ids=args.device_ids).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.95)
    fixed_z = torch.randn(args.sample_size, args.z_dim).to(device)
    criterion = VAELoss(args)

    for epoch in tqdm(range(args.n_epochs)):
        losses = []
        for img in loader:
            x = img.to(device)
            x_hat, mu, logvar = model(x)
            loss = criterion(x, x_hat, mu, logvar)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # logging & generating imgs from fixed vector
        writer.add_scalar('Loss', sum(losses) / len(losses), global_step=epoch)
        with torch.no_grad():
            sampled_images = model.sample(fixed_z)
        sampled_images = ((sampled_images + 1) / 2).view(-1, args.img_channels, args.img_size, args.img_size)
        writer.add_image('Generated Images', torchvision.utils.make_grid(sampled_images), global_step=epoch)
        tqdm.write(
            f'Epoch {epoch + 1}/{args.n_epochs}, \
                Loss: {sum(losses) / len(losses):.3f}'
        )
        torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, f"{args.checkpoint_dir}/VAE.pth")
