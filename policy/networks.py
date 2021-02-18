import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(True)
        )
        self.v = nn.Linear(hidden_dim // 2, 1)
        self.pi = nn.Linear(hidden_dim // 2, n_actions)

    def forward(self, state):
        features = self.encoder(state)
        return self.v(features), self.pi(features)
