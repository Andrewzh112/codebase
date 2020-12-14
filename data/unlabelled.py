from glob import glob
import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CelebA
import torch
from torch.utils.data import Dataset, DataLoader


class celebA(Dataset):
    def __init__(self, data_path, img_ext, crop_size, img_size):
        self.celebs = glob(data_path + '/*' + img_ext)
        Range = transforms.Lambda(lambda X: 2 * X - 1.)
        self.transforms = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Range
        ])

    def __len__(self):
        return len(self.celebs)

    def __getitem__(self, index):
        img = Image.open(self.celebs[index]).convert('RGB')
        return self.transforms(img)


def get_celeba_loaders(data_path, img_ext, crop_size, img_size, batch_size, download):
    dataset = celebA(data_path, img_ext, crop_size, img_size)
    if download:
        dataset = CelebA('celebA', transform=dataset.transforms, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=28)
