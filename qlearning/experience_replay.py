import torch


class ReplayBuffer:
    def __init__(self, max_size, state_dim, alpha=1):
        self.states = torch.empty(max_size, *state_dim)
        self.rewards = torch.empty(max_size)
        self.actions = torch.zeros(max_size, dtype=torch.long)
        self.next_states = torch.empty(max_size, *state_dim)
        self.dones = torch.ones(max_size, dtype=torch.bool)
        self.priorities = torch.zeros(max_size)
        self.max_size = max_size
        self.alpha = alpha
        self.ctr = 0

    def store(self, state, reward, action, next_state, done, priority=None):
        i = self.ctr % self.max_size
        self.states[i] = state.cpu()
        self.rewards[i] = reward
        self.actions[i] = torch.tensor(action, dtype=torch.long)
        self.next_states[i] = next_state.cpu()
        self.dones[i] = done
        if priority is not None:
            idx, priority = priority
            self.priorities[idx] = priority.cpu()
        else:
            self.priorities[i] = 1
        self.ctr += 1

    def sample(self, batch_size, device, beta=0):
        max_mem = min(self.ctr, self.max_size)
        assert max_mem > 0
        sample_distribution = self.priorities ** self.alpha
        sample_distribution /= sample_distribution.sum()
        idx = torch.multinomial(sample_distribution, batch_size)
        states = self.states[idx].to(device)
        rewards = self.rewards[idx].to(device)
        actions = self.actions[idx]
        next_states = self.next_states[idx].to(device)
        dones = self.dones[idx].to(device)
        weights = ((max_mem * sample_distribution[idx]) ** (- beta)).to(device)
        weights /= weights.max()
        return states, rewards, actions, next_states, dones, idx, weights
