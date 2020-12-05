import numpy as np
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from simsiam.model import SimSiam
from simsiam.main import args


def get_features(loader, model, device):
    features, targets = [], []
    for img, target in loader:
        img = img.to(device)
        with torch.no_grad():
            feature = model(img, istrain=False)
        targets.extend(target.cpu().numpy().tolist())
        features.append(feature.cpu())
    features = torch.cat(features).numpy()
    return features, targets


def scatter(class_dict, features, targets, savefig=True):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'violet', 'orange', 'purple']
    plt.figure(figsize=(12, 10))
    for class_, i in class_dict.items():
        idx = np.where(np.stack(targets) == i)[0]
        plt.scatter(features[idx, 0], features[idx, 1], c=colors[i], label=class_)
    plt.legend()
    if savefig:
        plt.savefig('simsiam/TSNE Features of SimSiam.jpg')
    plt.show()


def visualize():
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

    # gather features
    features, targets = get_features(loader, model, device)

    # get tsne features
    tsne = TSNE(n_components=2, random_state=0).fit_transform(features)

    # plot scatter
    scatter(data.class_to_idx, tsne, targets)

if __name__ == '__main__':
    visualize()
