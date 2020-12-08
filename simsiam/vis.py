import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from simsiam.model import SimSiam
from simsiam.main import args
from utils.contrastive import get_feature_label, tsne_visualize


if __name__ == '__main__':
    # perform tsne on test data
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
    data = CIFAR10(root=args.data_root, train=False, transform=transform, download=True)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=28)

    # load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(SimSiam(args), device_ids=args.device_ids).to(device)
    model.load_state_dict(torch.load(f"{args.check_point}"))
    model.eval()

    tsne_visualize(model, loader, device, data.class_to_idx, normalize=False, savefig='simsiam/tsne_features.jpg')
