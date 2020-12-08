import torch
from torch.nn import functional as F


def get_feature_label(feature_extractor, feature_loader, device, normalize=True, predictor=None):
    transformed_features, targets = [], []
    for data, target in feature_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            feature = feature_extractor(data)
        if normalize:
            feature = F.normalize(feature, dim=1)
        if predictor is None:
            transformed_features.append(feature.clone())
        else:
            transformed_features.append(predictor.predict(feature.clone()))
        targets.append(target)
    transformed_features = torch.cat(transformed_features, dim=0)
    targets = torch.cat(targets, dim=0)
    return transformed_features, targets
