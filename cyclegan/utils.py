from collections import deque
import numpy as np
import random
import torch
import torchvision


class LambdaLR:
    def __init__(self, n_epochs, starting_epoch, decay_epoch):
        self.n_epochs = n_epochs
        self.decay_epoch = decay_epoch
        self.starting_epoch = starting_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.starting_epoch - self.n_epochs)/(self.decay_epoch + 1)


class ReplayBuffer:
    def __init__(self, buffer_size=50):
        self.buffer = deque([], maxlen=buffer_size)

    def sample(self, in_batch):
        batch_size = in_batch.size(0)

        buffer_idx = np.array(range(len(self.buffer)))
        random.shuffle(buffer_idx)

        batch_idx = np.array(range(len(in_batch)))
        random.shuffle(batch_idx)
        if len(self.buffer) > 0:
            num_sample = np.random.binomial(batch_size, 0.5)
        else:
            num_sample = 0
        num_batch = batch_size - num_sample

        if len(buffer_idx[:num_sample]) > 0:
          buffer_samples = [sample for i, sample in enumerate(self.buffer) if i in buffer_idx[:num_sample]]
        else:
          buffer_samples = []
        if len(batch_idx[:num_batch]) > 0:
          batch_samples = in_batch[batch_idx[:num_batch]]
        else:
          batch_samples = []

        return_batch = torch.stack([*buffer_samples, *batch_samples], dim=0)
        self.buffer.extend([*in_batch])

        return return_batch[idx]


def make_images(image_tensor, size=(3, 224, 224)):
    image_tensor = torch.cat(image_tensor)
    image_tensor = (image_tensor + 1) / 2
    return torchvision.utils.make_grid(image_tensor.view(-1, *size))


def get_random_ids(idx_len, size):
    return np.random.choice(
            list(range(idx_len)),
            size=size,
            replace=False)
