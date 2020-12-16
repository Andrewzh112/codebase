import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.counter = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        state = self.state_memory[batch]
        state_ = self.new_state_memory[batch]
        done = self.terminal_memory[batch]
        reward = self.reward_memory[batch]
        action = self.action_memory[batch]

        return state, action, reward, state_, done
