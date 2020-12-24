import torch
import torchvision
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from networks.utils import initialize_modules
from data.unlabelled import get_celeba_loaders
from vae.model import VAE, VAE_Plus
from vae.loss import VAELoss, GroupSparsityLoss


parser = argparse.ArgumentParser()

# image settings
parser.add_argument('--img_channels', type=int, default=3, help='Numer of channels for images')
parser.add_argument('--img_size', type=int, default=64, help='H, W of the input images')
parser.add_argument('--crop_size', type=int, default=128, help='H, W of the input images')
parser.add_argument('--download', action="store_true", default=False, help='If auto download CelebA dataset')

# model params
parser.add_argument('--z_dim', type=float, default=100, help='dimension of random noise latent vector')
parser.add_argument('--n_res_blocks', type=int, default=1, help='Number of ResNet Blocks for generators')
parser.add_argument('--model_dim', type=float, default=128, help='model dimensions multiplier')
parser.add_argument('--plus', action="store_true", default=False, help='train modified vae')
parser.add_argument('--n_zelements', type=int, default=4, help='elements per gorup for z')
parser.add_argument('--n_pelements', type=int, default=128, help='elements per gorup for projected')

# loss fn
parser.add_argument('--beta', type=float, default=1., help='Beta hyperparam for KLD Loss')
parser.add_argument('--gamma', type=float, default=1., help='gamma hyperparam for Sparsity Loss')
parser.add_argument('--recon', type=str, default='bce', help='Reconstruction loss type [bce, l2]')

# training hyperparams
parser.add_argument('--device_ids', type=list, default=[0, 1], help='List of GPU devices')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for generators')
parser.add_argument('--betas', type=tuple, default=(0.5, 0.999), help='Betas for Adam optimizer')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')

# logging
parser.add_argument('--log_dir', type=str, default='vae/logs', help='Path to where log files will be saved')
parser.add_argument('--data_path', type=str, default='data/img_align_celeba', help='Path to where image data is located')
parser.add_argument('--sample_path', type=str, default='vae/samples', help='Path to where samples are saved')
parser.add_argument('--img_ext', type=str, default='.jpg', help='Image extentions')
parser.add_argument('--checkpoint_dir', type=str, default='vae/model_weights', help='Path to where model weights will be saved')

# for sampling
parser.add_argument('--sample_size', type=int, default=64, help='Size of sampled images')
parser.add_argument('--sample', action="store_true", default=False, help='Sample from VAE')
parser.add_argument('--walk', action="store_true", default=False, help='Walk through a feature & sample')

args = parser.parse_args()


if __name__ == '__main__':
    writer = SummaryWriter(args.log_dir + f'/{int(datetime.now().timestamp()*1e6)}')
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    loader = get_celeba_loaders(args.data_path, args.img_ext, args.crop_size,
                                args.img_size, args.batch_size, args.download)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize model, instantiate opt & scheduler & loss fn
    if args.plus:
        # model = torch.nn.DataParallel(
        #     VAE_Plus(args.z_dim, args.model_dim, args.img_size, args.img_channels),
        #     device_ids=args.device_ids).to(device)
        model = VAE_Plus(args.z_dim, args.model_dim, args.img_size, args.img_channels).to(device)
    else:
        model = torch.nn.DataParallel(
            VAE(args.z_dim, args.model_dim, args.img_size, args.img_channels, args.n_res_blocks),
            device_ids=args.device_ids).to(device)

    model.apply(initialize_modules)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.995)
    criterion = VAELoss(args.recon, args.beta)

    # fixed z to see how model changes on the same latent vectors
    fixed_z = torch.randn(args.sample_size, args.z_dim).to(device)

    pbar = tqdm(range(args.n_epochs))
    for epoch in pbar:
        losses, kdls, rls = [], [], []
        if args.plus:
            s_loss = []
        model.train()
        for img in loader:
            x = img.to(device)

            # x_hat for recon loss, mu & logvar for kdl loss
            if args.plus:
                x_hat, mu, logvar, z_p = model(x)
            else:
                x_hat, mu, logvar = model(x)

            # return kdl & recon loss for logging purposes
            loss, recon_loss, kld_loss = criterion(x, x_hat, mu, logvar)

            # if there is a sparsity loss
            if args.plus:
                std = logvar.mul(0.5).exp_()
                sparsity_loss = GroupSparsityLoss(args.n_zelements)(mu) + \
                    GroupSparsityLoss(args.n_zelements)(std) + \
                        GroupSparsityLoss(args.n_pelements)(z_p)
                loss += args.gamma * sparsity_loss
                s_loss.append(sparsity_loss.item())

            # logging and updating parameters
            losses.append(loss.item())
            kdls.append(kld_loss.item())
            rls.append(recon_loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.plus:
                pbar.set_postfix({'KLD Loss': kld_loss.item(),
                                'Reconstruction Loss': recon_loss.item(),
                                'Sparsity Loss': sparsity_loss.item()})
            else:
                pbar.set_postfix({'KLD Loss': kld_loss.item(),
                                'Reconstruction Loss': recon_loss.item()})
        scheduler.step()

        # logging & generating imgs from fixed vector
        writer.add_scalar('Loss', sum(losses) / len(losses), global_step=epoch)
        writer.add_scalar('KLD Loss', sum(kdls) / len(kdls), global_step=epoch)
        writer.add_scalar('Reconstruction Loss', sum(rls) / len(rls), global_step=epoch)
        if args.plus:
            writer.add_scalar('Sparsity Loss', sum(s_loss) / len(s_loss), global_step=epoch)

        # decode fixed z latent vectors
        model.eval()
        with torch.no_grad():
            sampled_images = model.module.sample(fixed_z)
            sampled_images = (sampled_images + 1) / 2

        # log images and losses & save model parameters
        writer.add_image('Fixed Generated Images', torchvision.utils.make_grid(sampled_images), global_step=epoch)
        writer.add_image('Reconstructed Images', torchvision.utils.make_grid(x_hat.detach()), global_step=epoch)
        writer.add_image('Original Images', torchvision.utils.make_grid(x.detach()), global_step=epoch)
        tqdm.write(
            f'Epoch {epoch + 1}/{args.n_epochs}, \
                Loss: {sum(losses) / len(losses):.3f}, \
                    Learning Rate: {round(scheduler.get_last_lr()[0], 4)}'
        )
        torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, f"{args.checkpoint_dir}/VAE.pth")
