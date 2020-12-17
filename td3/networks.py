import torch.nn as nn
import torch


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
