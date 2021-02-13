import numpy as np


class BlackJackAgent:
    def __init__(self, method, env, function='V', gamma=0.99, epsilon=0.1):
        self.method = method
        self.values = {(i, j, b): 0 for i in range(env.observation_space[0].n) for j in range(env.observation_space[1].n) for b in [True, False]}
        self.vreturns = {(i, j, b): [] for i in range(env.observation_space[0].n) for j in range(env.observation_space[1].n) for b in [True, False]}
        self.qs = {(i, j, b, a): 10 for i in range(env.observation_space[0].n) for j in range(env.observation_space[1].n) for b in [True, False] for a in range(env.action_space.n)}
        self.qreturns = {(i, j, b, a): [] for i in range(env.observation_space[0].n) for j in range(env.observation_space[1].n) for b in [True, False] for a in range(env.action_space.n)}
        self.value_function = lambda i, j, k: self.values[(i, j, k)]
        self.q_function = lambda i, j, k, l: self.qs[(i, j, k, l)]
        self.get_state_name = lambda state: (state[0], state[1], state[2])
        self.get_state_action_name = lambda state, action: (state[0], state[1], state[2], action)
        self.gamma = gamma
        self.actions = list(range(env.action_space.n))
        self.policy = {state: 0 for state in self.values.keys()}
        self.epsilon = epsilon
        self.function = function

    def choose_action(self, state):
        sum_, show, ace = state
        if self.method == 'lucky':
            return self.feeling_lucky(sum_)
        if self.method == 'egreedy':
            return self.epsilon_greedy(state)

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            state_name = self.get_state_name(state)
            return self.policy[state_name]

    def feeling_lucky(self, sum_):
        if sum_ < 20:
            return 1
        return 0

    def update(self, rewards, states, actions, function='V'):
        visited = set()
        if self.function == 'V':
            for i, state in enumerate(states):
                state_name = self.get_state_name(state)
                if state_name in visited:
                    continue
                G = 0
                for j, reward in enumerate(rewards[i:], 1):
                    G += self.gamma ** j * reward
                self.vreturns[state_name].append(G)
                self.values[state_name] = np.mean(self.vreturns[state_name])
                visited.add(state_name)
        elif self.function == 'Q':
            for i, (state, action) in enumerate(zip(states, actions)):
                state_action_name = self.get_state_action_name(state, action)
                if state_action_name in visited:
                    continue
                G = 0
                for j, reward in enumerate(rewards[i:], 1):
                    G += self.gamma ** j * reward
                self.qreturns[state_action_name].append(G)
                self.qs[state_action_name] = np.mean(self.qreturns[state_action_name])
                visited.add(state_action_name)
            for state in states:
                Q_prime, A_prime = -np.inf, None
                for action in actions:
                    state_action_name = self.get_state_action_name(state, action)
                    curr_Q = self.qs[state_action_name]
                    if curr_Q > Q_prime:
                        Q_prime = curr_Q
                        A_prime = action
                state_name = self.get_state_name(state)
                self.policy[state_name] = A_prime
        else:
            raise NotImplementedError


class CartPoleNoob:
    def __init__(self, method, env, function='V', alpha=0.1, gamma=0.99, epsilon=0.1, n_bins=10):
        self.method = method
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.function = function
        self.actions = list(range(env.action_space.n))
        self.rad = np.linspace(-0.2094, 0.2094, n_bins)
        self.values = {r: 0 for r in range(len(self.rad) + 1)}
        self.qs = {(r, a): 10 for r in range(len(self.rad) + 1) for a in self.actions}

    def choose_action(self, state):
        if self.method == 'naive':
            return self.naive_action(state)
        if self.method == 'egreedy':
            return self.epsilon_greedy(state)

    def naive_action(self, state):
        if state[2] < 0:
            return 0
        return 1

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            s = self.get_bucket_index([state[2]])[0]
            action = np.array([self.qs[(s, a)] for a in self.actions]).argmax()
            return action

    def get_bucket_index(self, states):
        inds = np.digitize(states, self.rad)
        return inds

    def update(self, state, action, reward, state_):
        r, r_ = self.get_bucket_index([state[2], state_[2]])
        if self.function == 'V':
            # TD update w/ bootstrap
            self.values[r] += self.alpha * (reward + self.gamma * self.values[r_] - self.values[r])
        elif self.function == 'Q':
            Q_ = np.array([self.qs[(r_, a)] for a in self.actions]).max()
            self.qs[(r, action)] += self.alpha * (reward + self.gamma * Q_ - self.qs[(r, action)])
            self.decrease_eps()

    def decrease_eps(self):
        self.epsilon = max(0.01, self.epsilon - 1e-5)
