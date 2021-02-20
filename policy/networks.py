import math
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


class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims, lr, weight_decay,
                 final_init, checkpoint_path, name):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.name = name
        encoder = []
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            encoder.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim)
            ])
            if i < len(hidden_dims) - 1:
                encoder.append(nn.ReLU(True))
            prev_dim = dim
        self.state_encoder = nn.Sequential(*encoder)
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, prev_dim),
                                            nn.LayerNorm(prev_dim))
        self.q = nn.Linear(prev_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self._init_weights(self.q, final_init)
        self._init_weights(self.action_encoder, 1 / math.sqrt(action_dim))
        self._init_weights(self.state_encoder, 1 / math.sqrt(hidden_dims[-2]))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_weights(self, layers, b):
        for m in layers.modules():
            if isinstance(m, (nn.Linear, nn.LayerNorm)):
                nn.init.uniform_(
                    m.weight,
                    a=-b,
                    b=b
                )

    def forward(self, states, actions):
        state_values = self.state_encoder(states)
        action_values = self.action_encoder(actions)
        state_action_values = nn.functional.relu(torch.add(state_values, action_values))
        return self.q(state_action_values)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path + '/' + self.name + '.pth')

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_path + '/' + self.name + '.pth'))


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims,
                 max_action, lr, weight_decay,
                 final_init, checkpoint_path, name):
        super().__init__()
        self.max_action = max_action
        self.name = name
        self.checkpoint_path = checkpoint_path
        encoder = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(True)])
            prev_dim = dim
        self.state_encoder = nn.Sequential(*encoder)
        self.mu = nn.Linear(prev_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self._init_weights(self.mu, final_init)
        self._init_weights(self.state_encoder, 1 / math.sqrt(hidden_dims[-2]))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_weights(self, layers, b):
        for m in layers.modules():
            if isinstance(m, (nn.Linear, nn.LayerNorm)):
                nn.init.uniform_(
                    m.weight,
                    a=-b,
                    b=b
                )

    def forward(self, states):
        state_features = self.state_encoder(states)
        # bound the output action to [-max_action, max_action]
        return torch.tanh(self.mu(state_features)) * self.max_action

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path + '/' + self.name + '.pth')

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_path + '/' + self.name + '.pth'))
