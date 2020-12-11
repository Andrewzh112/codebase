from torch import nn
import torch


class VAELoss(nn.Module):
    def __init__(self, recon, beta):
        super().__init__()
        if recon == 'l2':
            self.recon = nn.MSELoss(reduction='sum')
        elif recon == 'l1':
            self.recon = nn.L1Loss(reduction='sum')
        elif recon == 'bce':
            self.recon = nn.BCELoss(reduction='sum')
        else:
            raise NotImplementedError
        self.beta = beta

    def KLD_Loss(self, mu, logvar):
        KDL_batch = logvar.exp() - logvar - 1 + mu.pow(2)
        return self.beta / 2 * torch.sum(KDL_batch)

    def forward(self, x, x_hat, mu, logvar):
        reconstruction_loss = self.recon(x_hat, x)
        kld_loss = self.KLD_Loss(mu, logvar)
        return reconstruction_loss + kld_loss, reconstruction_loss, kld_loss
