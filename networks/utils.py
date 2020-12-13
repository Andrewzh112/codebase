from torch import nn
import torch


def initialize_modules(model, nonlinearity='leaky_relu'):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                m.weight,
                mode='fan_out',
                nonlinearity=nonlinearity
            )
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)


def load_weights(state_dict_path, models, model_names, optimizers=[], optimizer_names=[], return_val=None, return_vals=None):
    def put_in_list(item):
        if not isinstance(item, list, tuple) and item is not None:
            item = [item]
        return item

    model = put_in_list(models)
    model_names = put_in_list(model_names)
    optimizers = put_in_list(optimizers)
    optimizer_names = put_in_list(optimizer_names)
    return_vals = put_in_list(return_vals)

    state_dict = torch.load(state_dict_path)

    for model, model_name in zip(models, model_names):
        model.load_state_dict(state_dict[model_name])

    for optimizer, optimizer_name in zip(optimizers, optimizer_names):
        optimizer.load_state_dict(state_dict[optimizer_name])

    if return_val is not None:
        return state_dict[return_val]

    if return_vals is not None:
        return {key: state_dict[key] for key in return_vals}
