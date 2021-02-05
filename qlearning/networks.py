import torch.nn as nn
from torch.nn import functional as F
import torch
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


class QBasic(nn.Module):
    def __init__(self, input_channels, n_actions, cpt_dir, name,
                 img_size=84, hidden_dim=512, n_repeats=4, channels=[32, 64, 64],
                 kernel_sizes=[8, 4, 3], strides=[4, 2, 1]):
        super().__init__()
        q_network = []
        # CNN layers
        prev_ch = input_channels * n_repeats
        for ch, ks, sd in zip(channels, kernel_sizes, strides):
            q_network.append(nn.Conv2d(prev_ch, ch, kernel_size=ks, stride=sd))
            q_network.append(nn.ReLU())
            prev_ch = ch
        q_network.append(nn.Flatten())

        # find the feature dimension after CNN and flatten
        dummy_img = torch.empty(1, input_channels * n_repeats, img_size, img_size)
        fc_size = nn.Sequential(*q_network)(dummy_img).size(-1)

        # FC layers
        q_network.append(nn.Linear(fc_size, hidden_dim))
        q_network.append(nn.ReLU())
        q_network.append(nn.Linear(hidden_dim, n_actions))
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


class QDueling(nn.Module):
    def __init__(self, input_channels, n_actions, cpt_dir, name,
                 img_size=84, hidden_dim=512, n_repeats=4, channels=[32, 64, 64],
                 kernel_sizes=[8, 4, 3], strides=[4, 2, 1]):
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
        feature_extractor.append(nn.Linear(fc_size, hidden_dim))
        feature_extractor.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*feature_extractor)

        # value & advantage fns
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, n_actions)

        # training
        self.name = name
        self.n_actions = n_actions
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
