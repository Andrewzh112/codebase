import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from itertools import chain
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from vae.loss import VAELoss
from networks.layers import ConvNormAct
from networks.utils import initialize_modules


class Generator(nn.Module):
    def __init__(self, img_channels, w, img_size, z_dim):
        super().__init__()
        self.min_dim = img_size // (2**4)
        self.w = w

        self.project = nn.Linear(z_dim, w * 8 * self.min_dim ** 2, bias=False)
        self.bn = nn.BatchNorm2d(w * 8)
        self.up_block1 = ConvNormAct(w * 8, w * 4, mode='up')
        self.up_block2 = ConvNormAct(w * 4, w * 2, mode='up')
        self.up_block3 = ConvNormAct(w * 2, w, mode='up')
        self.up_block4 = nn.ConvTranspose2d(w, img_channels,
                                            kernel_size=4, stride=2,
                                            padding=1)

    def forward(self, x):
        batch_size =  x.size(0)
        x = self.project(x).reshape(batch_size, self.w * 128 // (self.min_dim ** 2), self.min_dim, self.min_dim)
        x = F.relu(self.bn(x))
        x = self.up_block1(x)
        x = self.up_block2(x)
        x = self.up_block3(x)
        x = self.up_block4(x)
        return torch.tanh(x)


class Critic(nn.Module):
    def __init__(self, img_channels, w, img_size):
        super().__init__()
        self.bn = nn.BatchNorm2d(w)
        self.down_block1 = nn.Conv2d(img_channels, w,
                                     kernel_size=3, stride=2,
                                     padding=1, bias=False)
        self.down_block2 = ConvNormAct(w, w * 2, mode='down', activation='lrelu')
        self.down_block3 = ConvNormAct(w * 2, w * 4, mode='down', activation='lrelu')
        self.down_block4 = ConvNormAct(w * 4, w * 8, mode='down', activation='lrelu')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w * 8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn(self.down_block1(x)), 0.2)
        x = self.down_block2(x)
        x = self.down_block3(x)
        x = self.down_block4(x)
        x = self.fc(self.avg_pool(x).squeeze())
        return x


class Decoder(Generator):
    def __init__(self, img_channels, w, img_size, z_dim):
        super().__init__(img_channels, w, img_size, z_dim)


class Encoder(Critic):
    def __init__(self, img_channels, w, img_size, z_dim):
        super().__init__(img_channels, w, img_size)
        self.fc = nn.Linear(w* 8, z_dim*2)
        self.z_dim = z_dim

    def forward(self, x):
        batch_size = x.size(0)
        x = F.leaky_relu(self.bn(self.down_block1(x)), 0.2)
        x = self.down_block2(x)
        x = self.down_block3(x)
        x = self.down_block4(x)
        x = self.fc(self.avg_pool(x).squeeze())
        x = x.view(batch_size, 2, self.z_dim)
        mu, logvar = x[:, 0, :], x[:, 1, :]
        return mu, logvar


class AVAE_Trainer(nn.Module):
    def __init__(self, img_channels, w, img_size, z_dim,
                 xi_dim, lr, betas, beta, epochs, sample_size,
                 logdir, checkpoint_dir, device):
        super().__init__()
        # models & optimizers
        self.G = Generator(img_channels, w, img_size, z_dim + xi_dim)
        self.C = Critic(img_channels, w, img_size)
        self.D = Decoder(img_channels, w, img_size, z_dim)
        self.E = Encoder(img_channels, w, img_size, xi_dim)
        self.optimizer_DE = torch.optim.Adam(
            chain(
                self.D.parameters(),
                self.E.parameters()
            ), lr=lr, betas=betas)
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=betas)
        self.optimizer_C = torch.optim.Adam(self.C.parameters(), lr=lr, betas=betas)

        # attributes
        self.xi_dim = xi_dim
        self.z_dim = z_dim

        # training attributes & logging
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(logdir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.epochs = epochs
        self.writer = SummaryWriter(logdir + f'/{int(datetime.now().timestamp()*1e6)}')
        self.vae_losses, self.gen_losses, self.critic_losses = [], [], []
        self.fixed_z = torch.randn(sample_size, z_dim, device=device)
        self.fixed_xi = torch.randn(sample_size, xi_dim, device=device)

        # criterions
        self.vae_criterion = VAELoss('bce', beta)
        self.critic_criterion = nn.BCEWithLogitsLoss()

        initialize_modules(self, init_type='kaiming')

    def gen_diff_criterion(self, mu, logvar, z):
        numerator = mu - (1 - logvar.exp()).sqrt() * z
        loss = 1 / 2 * torch.norm(numerator / logvar.mul(0.5).exp_(), p=2, dim=1) ** 2
        return loss.mean()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        return torch.randn(mu.size(), device=mu.device).mul(std).add_(mu)

    def train_step(self, x_real):
        x_real = x_real.to(self.device)

        # sample latent codes
        batch_size = x_real.size(0)
        xi = torch.randn(batch_size, self.xi_dim, device=x_real.device)
        z_fake = torch.randn(batch_size, self.z_dim, device=x_real.device)

        # VAE forward
        z_mu_real, z_logvar_real = self.E(x_real)
        z_real = self.reparameterize(z_mu_real, z_logvar_real)
        mu_real = self.D(z_real)

        # update VAE encoder & decoder
        self.optimizer_DE.zero_grad()
        vae_loss, _, _ = self.vae_criterion(
            x_real,
            mu_real,
            z_mu_real,
            z_logvar_real)
        vae_loss.backward()
        self.optimizer_DE.step()

        # GAN forward
        x_fake = self.G(torch.cat([z_fake, xi], dim=1))
        z_mu_fake, z_logvar_fake = self.E(x_fake)
        c_real, c_fake = self.C(x_real), self.C(x_fake.detach())

        # update GAN critic
        self.optimizer_C.zero_grad()
        critic_loss = self.critic_criterion(
            c_real,
            torch.ones_like(c_real)
        ) + self.critic_criterion(
            c_fake,
            c_fake
        )
        critic_loss.backward()
        self.optimizer_C.step()

        # update GAN generator
        self.optimizer_G.zero_grad()
        gen_loss = self.critic_criterion(
            self.C(self.G(torch.cat([z_fake, xi], 1))),
            torch.ones_like(c_fake)
        ) + self.gen_diff_criterion(
            z_mu_fake,
            z_logvar_fake,
            z_fake
        )
        gen_loss.backward()
        self.optimizer_G.step()

        return {
            'vae_loss': vae_loss.item(),
            'critic_loss': critic_loss.item(),
            'gen_loss': gen_loss.item()
        }

    def eval_step(self):
        with torch.no_grad():
            sampled_D = (self.D(self.fixed_z) + 1) / 2
            sampled_G = (self.G(torch.cat([self.fixed_z], 1)) + 1) / 2
        return {'sampled_D': sampled_D, 'sampled_G': sampled_G}

    def log_everything(self, items, pbar=None, step=None):
        if self.training:
            vae_loss, gen_loss, critic_loss = (
                items['vae_loss'],
                items['gen_loss'],
                items['critic_loss']
            )
            self.vae_losses.append(vae_loss)
            self.gen_losses.append(gen_loss)
            self.critic_losses.append(critic_loss)
            pbar.set_postfix({'VAE Loss': vae_loss,
                              'Critic Loss': critic_loss,
                              'Generator Loss': gen_loss})
        else:
            # logging epoch losses
            self.writer.add_scalar('VAE Loss', sum(self.vae_losses) / len(self.vae_losses), global_step=step)
            self.writer.add_scalar('Critic Loss', sum(self.critic_losses) / len(self.critic_losses), global_step=step)
            self.writer.add_scalar('Generator Loss', sum(self.gen_losses) / len(self.gen_losses), global_step=step)
            tqdm.write(
            f'Epoch {step + 1}/{self.epochs}, \
                VAE Loss: {sum(self.vae_losses) / len(self.vae_losses):.3f}, \
                    Critic Loss: {sum(self.critic_losses) / len(self.critic_losses):.3f}, \
                        Generator Loss: {sum(self.gen_losses) / len(self.gen_losses):.3f}'
            )
            self.vae_losses, self.gen_losses, self.critic_losses = [], [], []

            # logging samples
            self.writer.add_image('Fixed Generator Images', make_grid(items['sampled_G']), global_step=step)
            self.writer.add_image('Fixed Decoder Images', make_grid(items['sampled_D']), global_step=step)
    
    def save_everything(self):
        torch.save({
                    'G': self.G.state_dict(),
                    'C': self.C.state_dict(),
                    'D': self.D.state_dict(),
                    'E': self.E.state_dict(),
                    'optimizer_DE': self.optimizer_DE.state_dict(),
                    'optimizer_C': self.optimizer_C.state_dict(),
                    'optimizer_G': self.optimizer_G.state_dict(),
                }, f"{self.checkpoint_dir}/AVAE.pth")

    def fit(self, train_loader):
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            self.train()
            for imgs in train_loader:
                outputs = self.train_step(imgs)
                self.log_everything(outputs, pbar=pbar)

            self.eval()
            outputs = self.eval_step()
            self.log_everything(outputs, step=epoch)
            self.save_everything()
