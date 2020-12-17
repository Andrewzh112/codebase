import torch


def get_random_labels(num_classes, batch_size, device):
    return torch.randint(low=0, high=num_classes, size=(batch_size,),
                         dtype=torch.long, device=device)
