from torch import nn
import torch
from networks.layers import ConvNormAct, ResBlock, SA_Conv2d, SN_ConvTranspose2d


class VAE(nn.Module):
    def __init__(self, z_dim, model_dim, img_size, img_channels, n_res_blocks=0):
        super().__init__()
        self.z_dim = z_dim
        self.model_dim = model_dim
        self.img_size = img_size

        self.encoder = nn.Sequential(
            ConvNormAct(img_channels, model_dim, 'basic', 'down', activation='lrelu'),
            ConvNormAct(model_dim, model_dim * 2, 'basic', 'down', activation='lrelu'),
            ConvNormAct(model_dim * 2, model_dim * 4, 'basic', 'down', activation='lrelu'),
            ConvNormAct(model_dim * 4, model_dim * 8, 'basic', 'down', activation='lrelu'),
            *[ResBlock(model_dim * 8, 'relu', 'bn') for _ in range(n_res_blocks)],
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear((img_size // (2**4))**2 * model_dim * 8, z_dim * 2)
        )
        self.projector = nn.Sequential(
            nn.Linear(z_dim, (img_size // (2**4))**2 * model_dim * 8),
            nn.BatchNorm1d((img_size // (2**4))**2 * model_dim * 8),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            ConvNormAct(model_dim * 8, model_dim * 4, 'basic', 'up', activation='lrelu'),
            ConvNormAct(model_dim * 4, model_dim * 2, 'basic', 'up', activation='lrelu'),
            ConvNormAct(model_dim * 2, model_dim, 'basic', 'up', activation='lrelu'),
            nn.ConvTranspose2d(model_dim, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        return torch.randn(mu.size(), device=mu.device).mul(std).add_(mu)

    def forward(self, x):
        batch_size = x.size(0)
        z = self.encoder(x)
        z = z.view(-1, 2, self.z_dim)
        mu, logvar = z[:, 0, :], z[:, 1, :]
        z = self.reparameterize(mu, logvar)
        z = self.projector(z).view(batch_size, self.model_dim * 8,
                                   self.img_size // (2**4), self.img_size // (2**4))
        return self.decoder(z), mu, logvar

    def sample(self, z=None, num_samples=50):
        if z is None:
            z = torch.randn(num_samples, self.z_dim, device=self.projector.device)
        z = self.projector(z).view(-1, self.model_dim * 8,
                                   self.img_size // (2**4), self.img_size // (2**4))
        return self.decoder(z)


class VAE_Plus(VAE):
    def __init__(self, z_dim, model_dim, img_size, img_channels):
        super().__init__(z_dim, model_dim, img_size, img_channels)
        self.encoder = nn.Sequential(
            ConvNormAct(img_channels, model_dim, 'sn', 'down', activation='lrelu'),
            SA_Conv2d(model_dim),
            ConvNormAct(model_dim, model_dim * 2, 'sn', 'down', activation='lrelu'),
            ConvNormAct(model_dim * 2, model_dim * 4, 'sn', 'down', activation='lrelu'),
            ConvNormAct(model_dim * 4, model_dim * 8, 'sn', None, activation='lrelu'),
            ConvNormAct(model_dim * 8, model_dim * 8, 'sn', None, activation='lrelu'),
            nn.Flatten(),
            nn.Linear((img_size // (2**3))**2 * model_dim * 8, z_dim * 2)
        )
        self.projector = nn.Sequential(
            nn.Linear(z_dim, (img_size // (2**3))**2 * model_dim * 8),
            nn.BatchNorm1d((img_size // (2**3))**2 * model_dim * 8),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            ConvNormAct(model_dim * 8, model_dim * 8, 'sn', None, activation='lrelu'),
            ConvNormAct(model_dim * 8, model_dim * 4, 'sn', None, activation='lrelu'),
            ConvNormAct(model_dim * 4, model_dim * 2, 'sn', 'up', activation='lrelu'),
            ConvNormAct(model_dim * 2, model_dim, 'sn', 'up', activation='lrelu'),
            SA_Conv2d(model_dim),
            SN_ConvTranspose2d(in_channels=model_dim, out_channels=img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)
        z = self.encoder(x)
        z = z.view(-1, 2, self.z_dim)
        mu, logvar = z[:, 0, :], z[:, 1, :]
        z = self.reparameterize(mu, logvar)
        z_p = self.projector(z).view(batch_size, self.model_dim * 8,
                                   self.img_size // (2**3), self.img_size // (2**3))
        return self.decoder(z_p), mu, logvar, z_p
