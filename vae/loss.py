from torch import nn
import torch


class VAELoss(nn.Module):
    def __init__(self, recon=None, beta=0.5):
        super().__init__()
        if recon == 'l2':
            self.recon = nn.MSELoss()
        else:
            self.recon = nn.L1Loss()
        self.beta = beta

    def _KL_Loss(self, mu, logvar):
        return torch.mean(self.beta / 2 * torch.sum(logvar.exp() - logvar - 1 + mu ** 2, dim=1), dim=0)

    def forward(self, x, x_hat, mu, logvar):
        reconstruction_loss = self.recon(x_hat, x)
        kl_loss = self._KL_Loss(mu, logvar)
        return reconstruction_loss + kl_loss
