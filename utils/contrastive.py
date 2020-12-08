import torch
from torch.nn import functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_feature_label(feature_extractor, feature_loader, device, normalize=True, predictor=None):
    transformed_features, targets = [], []
    for data, target in feature_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            feature = feature_extractor(data)
            if normalize:
                feature = F.normalize(feature, dim=1)
        if predictor is None:
            transformed_features.append(feature)
        else:
            transformed_features.append(predictor.predict(feature))
        targets.append(target)
    transformed_features = torch.cat(transformed_features, dim=0)
    targets = torch.cat(targets, dim=0)
    return transformed_features, targets


def scatter(class_dict, features, targets, savefig=False):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'violet', 'orange', 'purple']
    colors = colors[:len(class_dict)]
    plt.figure(figsize=(12, 10))
    for class_, i in class_dict.items():
        idx = np.where(np.stack(targets) == i)[0]
        plt.scatter(features[idx, 0], features[idx, 1], c=colors[i], label=class_)
    plt.legend()
    if savefig:
        plt.savefig(savefig)
    plt.show()


def tsne_visualize(model, loader, device, classes, normalize=False, savefig=False):
    """Visualize extracted features using TSNE"""
    # gather features
    features, targets = get_feature_label(model, loader, device, normalize=normalize)
    features, targets = features.cpu().numpy(), targets.cpu().numpy()

    # get tsne features
    tsne = TSNE(n_components=2, random_state=0).fit_transform(features)

    # plot scatter
    scatter(classes, tsne, targets, savefig)
