from torch import nn
import torch
from networks.layers import ConvNormAct, ResBlock, Reshape


class VAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = nn.Sequential(
            ConvNormAct(args.img_channels, args.model_dim, 'down', activation='lrelu'),
            ConvNormAct(args.model_dim, args.model_dim * 2, 'down', activation='lrelu'),
            ConvNormAct(args.model_dim * 2, args.model_dim * 4, 'down', activation='lrelu'),
            ConvNormAct(args.model_dim * 4, args.model_dim * 8, 'down', activation='lrelu'),
            *[ResBlock(args.model_dim * 8, 'relu', 'bn') for _ in range(args.n_res_blocks)],
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear((args.img_size // (2**4))**2 * args.model_dim * 8, args.z_dim * 2)
        )
        self.projector = nn.Sequential(
            nn.Linear(args.z_dim, (args.img_size // (2**4))**2 * args.model_dim * 8),
            nn.BatchNorm1d((args.img_size // (2**4))**2 * args.model_dim * 8),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            ConvNormAct(args.model_dim * 8, args.model_dim * 4, 'up', activation='lrelu'),
            ConvNormAct(args.model_dim * 4, args.model_dim * 2, 'up', activation='lrelu'),
            ConvNormAct(args.model_dim * 2, args.model_dim, 'up', activation='lrelu'),
            ConvNormAct(args.model_dim, args.img_channels, 'up', activation='lrelu'),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        return torch.randn(mu.size(), device=mu.device).mul(std).add_(mu)

    def forward(self, x):
        batch_size = x.size(0)
        z = self.encoder(x)
        z = z.view(-1, 2, self.args.z_dim)
        mu, logvar = z[:, 0, :], z[:, 1, :]
        z = self.reparameterize(mu, logvar)
        z = self.projector(z).view(batch_size, self.args.model_dim * 8,
                                   self.args.img_size // (2**4), self.args.img_size // (2**4))
        return self.decoder(z), mu, logvar

    def sample(self, z=None, num_samples=50):
        if z is None:
            z = torch.randn(num_samples, self.args.z_dim, device=next(self.parameters()).device)
        z = self.projector(z).view(-1, self.args.model_dim * 8,
                                   self.args.img_size // (2**4), self.args.img_size // (2**4))
        return self.decoder(z)
