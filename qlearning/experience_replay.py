import torch


class ReplayBuffer:
    def __init__(self, max_size, state_dim):
        self.states = torch.empty(max_size, *state_dim)
        self.rewards = torch.empty(max_size)
        self.actions = torch.zeros(max_size, dtype=torch.long)
        self.next_states = torch.empty(max_size, *state_dim)
        self.dones = torch.ones(max_size, dtype=torch.bool)
        self.priorities = torch.zeros(max_size)
        self.max_size = max_size
        self.ctr = 0

    def store(self, state, reward, action, next_state, done, priority=None):
        i = self.ctr % self.max_size
        self.states[i] = state.cpu()
        self.rewards[i] = reward
        self.actions[i] = torch.tensor(action, dtype=torch.long)
        self.next_states[i] = next_state.cpu()
        self.dones[i] = done
        if priority is not None:
            self.priorities[i] = priority
        else:
            self.priorities[i] = 1
        self.ctr += 1

    def sample(self, batch_size, device):
        max_mem = min(self.ctr, self.max_size)
        assert max_mem > 0
        idx = torch.multinomial(self.priorities, batch_size)
        states = self.states[idx].to(device)
        rewards = self.rewards[idx].to(device)
        actions = self.actions[idx]
        next_states = self.next_states[idx].to(device)
        dones = self.dones[idx].to(device)
        return states, rewards, actions, next_states, dones
