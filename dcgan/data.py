from glob import glob
import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, random_split, DataLoader


class CelebA(Dataset):
    def __init__(self, args):
        self.celebs = glob(args.data_path + '/*' + args.img_ext)
        self.transforms = transforms.Compose([
            transforms.CenterCrop(args.crop_size),
            transforms.Resize(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.celebs)

    def __getitem__(self, index):
        img = Image.open(self.celebs[index]).convert('RGB')
        return self.transforms(img)


def get_loaders(args):
    dataset = CelebA(args)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
