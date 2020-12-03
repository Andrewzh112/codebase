from torch import nn
import torch


class VAELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.recon == 'l2':
            self.recon = nn.MSELoss(reduction='sum')
        elif args.recon == 'l1':
            self.recon = nn.L1Loss(reduction='sum')
        elif args.recon == 'bce':
            self.recon = nn.BCELoss(reduction='sum')
        else:
            raise NotImplementedError
        self.beta = args.beta

    def KLD_Loss(self, mu, logvar):
        KDL_batch = logvar.exp() - logvar - 1 + mu.pow(2)
        return self.beta / 2 * torch.sum(KDL_batch)

    def forward(self, x, x_hat, mu, logvar):
        reconstruction_loss = self.recon(x_hat, x)
        kld_loss = self.KLD_Loss(mu, logvar)
        return reconstruction_loss + kld_loss, reconstruction_loss, kld_loss
