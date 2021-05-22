import math
import numpy as np
import torch
from torch import nn
# from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


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

    def save_checkpoint(self, info):
        torch.save(self.state_dict(), self.checkpoint_path + '/' + info + '_' + self.name + '.pth')

    def load_checkpoint(self, info):
        self.load_state_dict(torch.load(self.checkpoint_path + '/' + info + '_' + self.name + '.pth'))


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

    def save_checkpoint(self, info):
        torch.save(self.state_dict(), self.checkpoint_path + '/' + info + '_' + self.name + '.pth')

    def load_checkpoint(self, info):
        self.load_state_dict(torch.load(self.checkpoint_path + '/' + info + '_' + self.name + '.pth'))


class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, lr,
                checkpoint_path, name):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.name = name
        encoder = []
        prev_dim = state_dim + action_dim
        for i, dim in enumerate(hidden_dims):
            encoder.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim)
            ])
            if i < len(hidden_dims) - 1:
                encoder.append(nn.ReLU(True))
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder)
        self.value = nn.Linear(prev_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, states, actions):
        scores = self.encoder(torch.cat([states, actions], dim=1))
        return self.value(scores)

    def save_checkpoint(self, info):
        torch.save(self.state_dict(), self.checkpoint_path + '/' + info + '_' + self.name + '.pth')

    def load_checkpoint(self, info):
        self.load_state_dict(torch.load(self.checkpoint_path + '/' + info + '_' + self.name + '.pth'))


class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_dims, log_std_min, log_std_max, epsilon,
                 lr, max_action, checkpoint_path, name):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon
        self.checkpoint_path = checkpoint_path
        self.name = name
        self.max_action = max_action
        encoder = []
        prev_dim = state_dim
        for i, dim in enumerate(hidden_dims):
            encoder.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim)
            ])
            if i < len(hidden_dims) - 1:
                encoder.append(nn.ReLU(True))
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder)

        # mu & logvar for action
        self.actor = nn.Linear(prev_dim, action_dim * 2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.target_entropy = -np.prod(action_dim)
        self.logalpha = nn.Parameter(torch.zeros(1))
        self.to(self.device)

    def sample(self, mu, log_std):
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
        # distribution = MultivariateNormal(mu, covariance_matrix=torch.diag_embed(log_std.exp()))
        distribution = Normal(mu, log_std.exp())
        actions = distribution.rsample()
        log_probs = distribution.log_prob(actions).sum(-1)
        normalized_actions = torch.tanh(actions)
        bounded_actions = normalized_actions * self.max_action
        bounded_log_probs = log_probs - (2 * (
            np.log(2) - actions - F.softplus(-2 * actions))).sum(axis=1)
        return bounded_actions.squeeze(), bounded_log_probs

    def forward(self, states):
        if states.dim() == 1:
            states = states.unsqueeze(0)
        scores = self.encoder(states)
        mu, log_std = self.actor(scores).chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        action, log_prob = self.sample(mu, log_std)
        return action, log_prob

    def save_checkpoint(self, info):
        torch.save(self.state_dict(), self.checkpoint_path + '/' + info + '_' + self.name + '.pth')

    def load_checkpoint(self, info):
        self.load_state_dict(torch.load(self.checkpoint_path + '/' + info + '_' + self.name + '.pth'))


class PPOActor(SACActor):
    def __init__(self, state_dim, action_dim,
                 hidden_dims, log_std_min, log_std_max, epsilon,
                 lr, max_action, checkpoint_path, name):
        super().__init__(state_dim, action_dim,
                         hidden_dims, log_std_min, log_std_max, epsilon,
                         lr, max_action, checkpoint_path, name)


class PPOCritic(SACActor):
    def __init__(self, state_dim, hidden_dims, lr,
                checkpoint_path, name):
        super().__init__(state_dim, 0, hidden_dims, lr, checkpoint_path, name)

    def forward(self, states):
        return self.value(self.encoder(states))
