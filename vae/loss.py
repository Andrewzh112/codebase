from torch import nn
import torch


class VAELoss(nn.Module):
    def __init__(self, recon=None, beta=1.):
        super().__init__()
        if recon == 'l2':
            self.recon = nn.MSELoss()
        else:
            self.recon = nn.BCELoss(reduction='sum')
        self.beta = beta

    def _KL_Loss(self, mu, logvar):
        KDL_batch = mu ** 2 + (logvar - logvar.exp() + 1)
        return self.beta / 2 * torch.sum(KDL_batch, dim=0)

    def forward(self, x, x_hat, mu, logvar):
        reconstruction_loss = self.recon(x_hat, x)
        kl_loss = self._KL_Loss(mu, logvar)
        return reconstruction_loss + kl_loss
