import pandas as pd
import numpy as np


class Tabular_Agent:
    def __init__(self, states, actions, epsilon_init, epsilon_min, epsilon_desc, gamma, alpha, n_episodes):
        self.actions = actions
        self.n_actions = len(actions)
        self.states = states
        self.n_states = len(states)
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_desc = epsilon_desc
        self.gamma = gamma
        self.alpha = alpha
        self.n_episodes = n_episodes
        # initialize table with 0 Q-values
        self.q_table = pd.DataFrame(np.zeros((self.n_states, self.n_actions)),
                                    index=states, columns=actions)

    def greedy_action(self, state):
        Qs = self.q_table.loc[state]
        return Qs.argmax()

    def strategize(self, state):
        if np.random.random() > self.epsilon:
            return self.greedy_action(state)
        return np.random.choice(self.actions)

    def update(self, state, action, reward, next_state):
        # update Q-table
        max_Q_ = self.q_table.loc[next_state].max()
        Q_sa = self.q_table.loc[state, action]
        self.q_table.loc[state, action] += self.alpha * (
            reward + self.gamma * max_Q_ - Q_sa)
        # update epsilon
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_desc)
