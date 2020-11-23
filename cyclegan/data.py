from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import os
from PIL import Image
import random


class CycleImageDataset(Dataset):
    def __init__(self, root, data_type, load_shape, target_shape):
        super().__init__()
        self.transforms = transforms.Compose([
            transforms.Resize(load_shape),
            transforms.RandomCrop(target_shape),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.imagesA = glob(os.path.join(root, '%sA' % data_type) + '/*.*')
        self.imagesB = glob(os.path.join(root, '%sB' % data_type) + '/*.*')
        self.extras = self.imagesA if len(self.imagesA) > len(self.imagesB) else self.imagesB

    def _shuffle(self):
        random.shuffle(self.extras)

    def __getitem__(self, index):
        imageA = self.transforms(Image.open(self.imagesA[index]).convert("RGB"))
        imageB = self.transforms(Image.open(self.imagesB[index]).convert("RGB"))
        if index == len(self) - 1:
            self._shuffle()
        return imageA, imageB

    def __len__(self):
        return min(len(self.imagesA), len(self.imagesB))
