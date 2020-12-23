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
        self.order = order

        convs = []
        prev_ch, ch = in_channels, multiplier
        for i in range(depth):
            if i == depth - 1:
                convs.append(nn.Conv2d(in_channels=prev_ch, out_channels=ch, kernel_size=4, padding=1, stride=2))
            else:
                convs.append(ConvNormAct(in_channels=prev_ch, out_channels=ch, mode='down'))
                prev_ch = ch
                ch *= 2
        self.convs = nn.Sequential(
            *convs,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten())
        self.fc = nn.Linear(order*ch, n_actions)

    def forward(self, imgs):
        img_feature = [self.convs(imgs[:, i*self.order:(i+1)*self.order, :, :]) for i in range(self.order)]
        img_feature = torch.cat(img_feature, 1)
        return torch.tanh(self.fc(img_feature)) * self.max_action


class ImageCritic(nn.Module):
    def __init__(self, in_channels, n_actions, hidden_dim, action_embed_dim, order, depth, multiplier, name):
        super().__init__()
        self.name = name
        self.order = order

        # constructed simple cnn
        convs = []
        prev_ch, ch = in_channels, multiplier
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
            nn.Linear(ch*order + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        img_embedding = [self.avg_pool(self.convs(
            state[:, i*self.order:(i+1)*self.order, :, :])).squeeze() for i in range(self.order)]
        img_embedding = torch.cat(img_embedding, 1)
        action_embedding = self.action_head(action)
        combined_embedding = torch.cat([img_embedding, action_embedding], dim=1)
        return self.combined_critic_head(combined_embedding)
