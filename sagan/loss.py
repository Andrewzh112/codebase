from torch import nn
import torch
from torch.autograd import Variable


class Hinge_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        assert reduction in ('sum', 'mean')
        self.reduction = reduction

    def forward(self, fake_logits, mode, real_logits=None):
        assert mode in ('generator', 'discriminator')
        if mode == 'generator':
            return self._generator_loss(fake_logits)
        return self._discriminator_loss(real_logits, fake_logits)
    
    def _reduct_loss(self, loss):
        if self.reduction == 'mean':
            return loss.mean(0)
        elif self.reduction == 'sum':
            return loss.sum(0)

    def _generator_loss(self, fake_logits):
        return self._reduct_loss(-fake_logits)

    def _discriminator_loss(self, real_logits, fake_logits):
        loss = torch.clamp(1 - real_logits, 0) + torch.clamp(1 + fake_logits, 0)
        return self._reduct_loss(loss)


class Wasserstein_GP_Loss(nn.Module):
    def __init__(self, lambda_gp=10, reduction='mean'):
        super().__init__()
        assert reduction in ('sum', 'mean')
        self.reduction = reduction
        self.lambda_gp = lambda_gp

    def forward(self, fake_logits, mode, real_logits=None):
        assert mode in ('generator', 'discriminator')
        if mode == 'generator':
            return self._generator_loss(fake_logits)
        elif mode == 'discriminator':
            return self._discriminator_loss(real_logits, fake_logits)

    def _generator_loss(self, fake_logits):
        return - fake_logits.mean()

    def _discriminator_loss(self, real_logits, fake_logits):
        return - real_logits.mean() + fake_logits.mean()

    def get_interpolates(self, reals, fakes):
        alpha = torch.rand(reals.size(0), 1, 1, 1).expand_as(reals).to(reals.device)
        interpolates = alpha * reals.data + ((1 - alpha) * fakes.data)
        return Variable(interpolates, requires_grad=True)

    def grad_penalty_loss(self, interpolates, interpolate_logits):
        gradients = torch.autograd.grad(outputs=interpolate_logits,
                                        inputs=interpolates,
                                        grad_outputs=interpolate_logits.new_ones(interpolate_logits.size()),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        return gradient_penalty
