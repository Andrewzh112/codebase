from torch import nn
import torch
from networks.layers import ConvNormAct, ResBlock, Reshape


class VAE(nn.Module):
    def __init__(self, args):
        self.args = args
        self.encoder = nn.Sequential(
            ConvNormAct(args.img_channels, args.model_dim, 'down'),
            ConvNormAct(args.model_dim, args.model_dim * 2, 'down'),
            ConvNormAct(args.model_dim * 2, args.model_dim * 4, 'down'),
            ConvNormAct(args.model_dim * 4, args.model_dim * 8, 'down'),
            *[ResBlock(args.model_dim * 8, 'relu', 'bn') for _ in range(args.n_res_blocks)],
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((args.img_size // (2**4))**2 * args.model_dim * 8, args.z_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.z_dim, (args.img_size // (2**4))**2 * args.model_dim * 8),
            nn.BatchNorm1d((args.img_size // (2**4))**2 * args.model_dim * 8),
            nn.ReLU(),
            Reshape(
                args.batch_size,
                args.model_dim * 8,
                args.img_size // (2**4),
                args.img_size // (2**4)
            ),
            ConvNormAct(args.model_dim * 8, args.model_dim * 4, 'up'),
            ConvNormAct(args.model_dim * 4, args.model_dim * 2, 'up'),
            ConvNormAct(args.model_dim * 2, args.model_dim, 'up'),
            ConvNormAct(args.model_dim, args.img_channels, 'up'),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        batch_size = mu.size(0)
        z = torch.randn(batch_size, self.args.z_dim) * torch.sqrt(torch.exp(logvar)) + mu
        return z

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(-1, 2, self.args.z_dim)
        mu, logvar = z[:, 0, :], z[:, 1, :]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def sample(self, z=None, num_samples=50):
        if z is None:
            z = torch.randn(num_samples, self.args.z_dim)
        return self.decoder(z)
