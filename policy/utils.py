import numpy as np
import torch


class OUActionNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.device = self.mu.device
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * torch.randn(size=self.mu.shape, device=self.device)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else torch.zeros_like(self.mu)


class GaussianActionNoise:
    def __init__(self, mu, sigma=0.2):
        self.mu = mu
        self.sigma = sigma
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __call__(self, output_dim, clip=None, sigma=None):
        if sigma is None:
            sigma = self.sigma
        noise = torch.randn(*output_dim) * sigma + self.mu
        if clip is not None:
            noise.clip(-clip, clip)
        return noise


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, maxsize):
        self.states = torch.empty(maxsize, *state_dim)
        self.actions = torch.empty(maxsize, *action_dim)
        self.next_states = torch.empty(maxsize, *state_dim)
        self.rewards = torch.empty(maxsize)
        self.dones = torch.zeros(maxsize, dtype=torch.bool)
        self.maxsize = maxsize
        self.idx = 0

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.idx % self.maxsize
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.idx += 1

    def sample_transition(self, batch_size):
        curr_size = min(self.maxsize, self.idx)
        idx = torch.multinomial(torch.ones(curr_size), batch_size)
        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        next_states = self.next_states[idx]
        dones = self.dones[idx]
        return states, actions, rewards, next_states, dones


def clip_action(action, max_action):
    return np.clip(action, - max_action, max_action)
