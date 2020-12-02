from torch import nn
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import ImageFilter, Image
from copy import deepcopy
import random


class MoCoLoss(nn.Module):
    def __init__(self, T=0.07):
        super().__init__()
        self.T = T
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, q, k, memo_bank):
        k = k.detach()
        pos_logits = torch.einsum('ij,ij->i', [q, k]).unsqueeze(-1)
        neg_logits = torch.einsum('ij,jk->ik', [q, memo_bank.queue.clone()])
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros_like(logits, dtype=torch.long, device=logits.device)
        return self.criterion(logits / self.T, labels)


class CIFAR10Pairs(CIFAR10):
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MemoryBank:
    """https://github.com/peisuke/MomentumContrast.pytorch"""
    def __init__(self, model_k, device, loader, K=4096):
        self.K = K
        self.queue = torch.zeros((0, 128), dtype=torch.float) 
        self.queue = self.queue.to(device)

        for data, _ in loader:
            x_k = data.to(device)
            k = model_k(x_k)
            k = k.detach()
            self.queue = self.queue_data(k)
            self.queue = self.dequeue_data(10)
            break

    def queue_data(self, k):
        k = k.detach()
        return torch.cat([self.queue, k], dim=0)

    def dequeue_data(self, K=None):
        if K is None:
            K = self.K
        assert isinstance(K, int)
        if len(self.queue) > K:
            return self.queue[-K:]
        else:
            return self.queue

    def dequeue_and_enqueue(self, k):
        self.queue_data(k)
        self.queue = self.dequeue_data()


def momentum_update(f_k, f_q, m):
    for param_q, param_k in zip(f_q.parameters(), f_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)


def get_momentum_encoder(f_q):
    f_k = deepcopy(f_q)
    for param in f_k.parameters():
        param.requires_grad = False
    return f_k
