from torch import nn
import torch


class VAELoss(nn.Module):
    def __init__(self, recon_type, beta, reduction='sum'):
        super().__init__()
        self.recon_type = recon_type
        if recon_type == 'l2':
            self.recon = nn.MSELoss(reduction=reduction)
        elif recon_type == 'l1':
            self.recon = nn.L1Loss(reduction=reduction)
        elif recon_type == 'bce':
            self.recon = nn.BCELoss(reduction=reduction)
        else:
            raise NotImplementedError
        self.beta = beta

    def KLD_Loss(self, mu, logvar):
        KDL_batch = logvar.exp() - logvar - 1 + mu.pow(2)
        return self.beta / 2 * torch.sum(KDL_batch)

    def forward(self, x, x_hat, mu, logvar):
        if self.recon_type == 'bce':
            x, x_hat = (x + 1) / 2, (x_hat + 1) / 2
        reconstruction_loss = self.recon(x_hat, x)
        kld_loss = self.KLD_Loss(mu, logvar)
        return reconstruction_loss + kld_loss, reconstruction_loss, kld_loss


class GroupSparsityLoss(nn.Module):
    def __init__(self, n_elements, rho=0.05):
        super().__init__()
        self.n_elements = n_elements
        self.rho = torch.tensor(rho)

    def forward(self, z):
        rho, n_elements = self.rho, self.n_elements
        # flatten latent variables & get dims
        z = z.view(z.size(0), -1)
        batch_size, z_dim = z.size()
        assert z_dim % n_elements == 0
        groups = z_dim // n_elements

        # apply mask to extract groups
        mask = torch.block_diag(
            *[torch.ones(n_elements) for i in range(groups)]
        ).unsqueeze(0).repeat(batch_size, 1, 1).to(z.device)
        z_groups = z.unsqueeze(1).repeat(1, groups, 1)
        masked_z = z_groups * mask
        rho_hat = masked_z.norm(p=2, dim=-1)

        sparsity_penalty = (
            (rho * (torch.log(rho) - torch.log(rho_hat))) + (
                    (1 - rho) * (torch.log(1 - rho) - torch.log(1 - rho_hat)))).sum(-1)
        return sparsity_penalty.mean()
