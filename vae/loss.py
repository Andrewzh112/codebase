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

    def _KLD_Loss(self, mu, logvar):
        KDL_batch = logvar.exp() - logvar - 1 + mu.pow(2)
        return self.beta / 2 * torch.sum(KDL_batch)

    def forward(self, x, x_hat, mu, logvar):
        reconstruction_loss = self.recon(x_hat, x)
        kld_loss = self._KLD_Loss(mu, logvar)
        return reconstruction_loss + kld_loss
