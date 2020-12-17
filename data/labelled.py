from torch.utils.data import Dataset, DataLoader
import torchvision


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.size(0)


def get_cifar_loader(batch_size, crop_size, img_size):
    Range = torchvision.transforms.Lambda(lambda X: 2 * X - 1.)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        Range
    ])
    dataset = torchvision.datasets.CIFAR10(root='.', train=True, transform=transforms, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), len(dataset.classes)
