from torch import nn
import torch


class VAELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.recon == 'l1':
            self.recon = nn.L1Loss()
        elif args.recon == 'l2':
            self.recon = nn.MSELoss()

    def _KL_Loss(self, mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

    def forward(self, x, x_hat, mu, logvar):
        reconstruction_loss = self.recon(x_hat, x)
        kl_loss = self._KL_Loss(mu, logvar)
        return reconstruction_loss + kl_loss
