from torch import nn
import torch


class SAGAN_Hinge_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, fake_logits, mode, real_logits=None):
        assert mode in ('generator', 'discriminator')
        if mode == 'generator':
            return self._generator_loss(fake_logits)
        return self._discriminator_loss(real_logits, fake_logits)

    def _generator_loss(self, fake_logits):
        if self.reduction == 'mean':
            return -fake_logits.mean(0)
        elif self.reduction == 'sum':
            return -fake_logits.sum(0)

    def _discriminator_loss(self, real_logits, fake_logits):
        loss = -torch.clamp(-1+real_logits, 0) - torch.clamp(-1 - fake_logits, 0)
        if self.reduction == 'mean':
            return loss.mean(0)
        elif self.reduction == 'sum':
            return loss.sum(0)
