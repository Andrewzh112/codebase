import torch.nn as nn
from torch.nn import functional as F
import torch
import math
import os


class QNaive(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, n_actions):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.state_embedder = nn.Linear(1, hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        state_emd = F.relu(self.state_embedder(state.float()))
        return self.fc(state_emd)


class DQN(nn.Module):
    def __init__(self, input_channels, out_features, cpt_dir, name,
                 img_size=84, hidden_dim=512, n_repeats=4, channels=[32, 64, 64],
                 kernel_sizes=[8, 4, 3], strides=[4, 2, 1], noised=False, **kwargs):
        super().__init__()
        feature_extractor = []
        # CNN layers
        prev_ch = input_channels * n_repeats
        for ch, ks, sd in zip(channels, kernel_sizes, strides):
            feature_extractor.append(nn.Conv2d(prev_ch, ch, kernel_size=ks, stride=sd))
            feature_extractor.append(nn.ReLU())
            prev_ch = ch
        feature_extractor.append(nn.Flatten())

        self.feature_extractor = nn.Sequential(*feature_extractor)
        q_network = [self.feature_extractor]

        # find the feature dimension after CNN and flatten
        dummy_img = torch.empty(1, input_channels * n_repeats, img_size, img_size)
        self.fc_size = self.feature_extractor(dummy_img).size(-1)

        # FC layers
        if noised:
            q_network.extend(
                [NoisedLinear(self.fc_size, hidden_dim), nn.ReLU(), NoisedLinear(hidden_dim, out_features)])
        else:
            q_network.extend(
                [nn.Linear(self.fc_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_features)])
        self.q_network = nn.Sequential(*q_network)

        # training
        self.name = name
        self.cpt = os.path.join(cpt_dir, name)

    def forward(self, observations):
        Qs = self.q_network(observations)
        return Qs

    def check_point(self):
        torch.save(self.state_dict(), self.cpt + '.pth')

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.cpt + '.pth'))


class DuelingDQN(nn.Module):
    def __init__(self, input_channels, out_features, cpt_dir, name,
                 img_size=84, hidden_dim=512, n_repeats=4, channels=[32, 64, 64],
                 kernel_sizes=[8, 4, 3], strides=[4, 2, 1], noised=False, **kwargs):
        super().__init__()
        feature_extractor = []
        # CNN layers
        prev_ch = input_channels * n_repeats
        for ch, ks, sd in zip(channels, kernel_sizes, strides):
            feature_extractor.append(nn.Conv2d(prev_ch, ch, kernel_size=ks, stride=sd))
            feature_extractor.append(nn.ReLU())
            prev_ch = ch
        feature_extractor.append(nn.Flatten())

        # find the feature dimension after CNN and flatten
        dummy_img = torch.empty(1, input_channels * n_repeats, img_size, img_size)
        fc_size = nn.Sequential(*feature_extractor)(dummy_img).size(-1)

        # FC layers
        if noised:
            feature_extractor.append(NoisedLinear(fc_size, hidden_dim))
        else:
            feature_extractor.append(nn.Linear(fc_size, hidden_dim))
        feature_extractor.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*feature_extractor)

        # value & advantage fns
        if noised:
            self.value = NoisedLinear(hidden_dim, 1)
            self.advantage = NoisedLinear(hidden_dim, out_features)
        else:
            self.value = nn.Linear(hidden_dim, 1)
            self.advantage = nn.Linear(hidden_dim, out_features)

        # training
        self.name = name
        self.cpt = os.path.join(cpt_dir, name)

    def forward(self, observations):
        features = self.feature_extractor(observations)
        Vs = self.value(features)
        As = self.advantage(features)
        Qs = Vs + (As - torch.mean(As, -1, True))
        return Qs

    def check_point(self):
        torch.save(self.state_dict(), self.cpt + '.pth')

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.cpt + '.pth'))


class NoisedMatrix(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.matrix_mu = nn.Parameter(torch.randn(in_features, out_features))
        self.matrix_sigma = nn.Parameter(torch.randn(in_features, out_features))
        self.matrix_epsilon = torch.empty(in_features, out_features)

        # initialization scheme
        self.sigma_init = sigma_init
        init_range = math.sqrt(3 / in_features)
        self.init_weights(init_range)

    def assemble_parameters(self):
        self.reset_epsilon()
        return self.matrix_mu + self.matrix_sigma * self.matrix_epsilon

    def reset_epsilon(self):
        self.matrix_epsilon.normal_(0, 1)
        self.matrix_epsilon = self.matrix_epsilon.to(self.matrix_mu.device)

    def init_weights(self, init_range):
        self.matrix_mu.data = torch.nn.init.uniform_(
            self.matrix_mu.data, a=-init_range, b=init_range)
        self.matrix_sigma.data = torch.ones_like(self.matrix_sigma.data) * self.sigma_init

    def forward(self):
        pass


class NoisedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = NoisedMatrix(in_features, out_features)
        self.bias = NoisedMatrix(out_features, 1)

    def forward(self, state):
        if self.training:
            weight = self.weight.assemble_parameters()
            bias = self.bias.assemble_parameters().squeeze()
        else:
            weight = self.weight.matrix_mu
            bias = self.bias.matrix_mu.squeeze()
        Qs = state @ weight + bias
        return Qs

class CategoricalDQN(DQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        categorical_network = [self.feature_extractor]
        if self.noised:
            linear_layer = NoisedLinear
        else:
            linear_layer = nn.Linear
        categorical_network.extend(
            [linear_layer(self.fc_size, self.hidden_dim),
             nn.ReLU(),
             linear_layer(self.hidden_dim, self.n_actions * self.num_atoms)])
        self.categorical_network = nn.Sequential(*categorical_network)

    def forward(self, state):
        logits = self.categorical_network(state).view(-1, self.num_atoms)
        return torch.softmax(logits, dim=-1).view(-1, self.n_actions, self.num_atoms)
