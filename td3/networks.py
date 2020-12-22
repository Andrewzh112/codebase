import torch.nn as nn
import torch
from networks.layers import ConvNormAct


class Critic(nn.Module):
    def __init__(self, input_dims, hidden_dims, n_actions, name):
        super().__init__()
        self.name = name

        fcs = []
        prev_dim = input_dims + n_actions
        # input layers
        for hidden_dim in hidden_dims:
            fcs.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim

        # output layer
        fcs.append(nn.Linear(prev_dim, 1))
        self.q = nn.Sequential(*fcs)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q(x)


class Actor(nn.Module):
    def __init__(self, input_dims, hidden_dims, n_actions, max_action, name):
        super().__init__()
        self.name = name
        self.max_action = max_action

        fcs = []
        prev_dim = input_dims
        # input layers
        for hidden_size in hidden_dims:
            fcs.extend([nn.Linear(prev_dim, hidden_size), nn.ReLU()])
            prev_dim = hidden_size

        # output layer
        fcs.extend([nn.Linear(prev_dim, n_actions), nn.Tanh()])
        self.pi = nn.Sequential(*fcs)

    def forward(self, state):
        return self.max_action * self.pi(state)


class ImageActor(nn.Module):
    def __init__(self, in_channels, n_actions, max_action, order, depth, multiplier, name):
        super().__init__()
        self.name = name
        self.max_action = max_action

        convs = []
        prev_ch = in_channels*order
        ch = multiplier
        for i in range(depth):
            if i == depth - 1:
                convs.append(nn.Conv2d(in_channels=prev_ch, out_channels=ch, kernel_size=4, padding=1, stride=2))
            else:
                convs.append(ConvNormAct(in_channels=prev_ch, out_channels=ch, mode='down'))
                prev_ch = ch
                ch *= 2
        self.actor = nn.Sequential(
            *convs,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, n_actions),
            nn.Tanh()
        )

    def forward(self, imgs):
        return self.actor(imgs) * self.max_action


class ImageCritic(nn.Module):
    def __init__(self, in_channels, n_actions, hidden_dim, action_embed_dim, order, depth, multiplier, name):
        super().__init__()
        self.name = name

        # constructed simple cnn
        convs = []
        prev_ch = in_channels*order
        ch = multiplier
        for i in range(depth):
            if i == depth - 1:
                convs.append(nn.Conv2d(in_channels=prev_ch, out_channels=ch, kernel_size=4, padding=1, stride=2))
            else:
                convs.append(ConvNormAct(in_channels=prev_ch, out_channels=ch, mode='down'))
                prev_ch = ch
                ch *= 2
        self.convs = nn.Sequential(*convs)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # embed actions, concat w/ img and output critic
        self.action_head = nn.Sequential(
            nn.Linear(n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_embed_dim)
        )
        self.combined_critic_head = nn.Sequential(
            nn.Linear(ch + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        img_embedding = self.avg_pool(self.convs(state)).squeeze()
        action_embedding = self.action_head(action)
        combined_embedding = torch.cat([img_embedding, action_embedding], dim=1)
        return self.combined_critic_head(combined_embedding)
