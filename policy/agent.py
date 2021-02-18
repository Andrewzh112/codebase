import numpy as np
import torch
from policy.networks import ActorCritic


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


class PolicyGradientAgent:
    def __init__(self, input_dim, action_dim, hidden_dim, gamma, lr):
        self.gamma = gamma
        self.policy = ActorCritic(*input_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.reward_history, self.action_logprob_history = [], []

    def choose_action(self, state):
        state = torch.from_numpy(state).to(self.device)
        action_proba = torch.softmax(self.policy(state), dim=-1)
        action_dist = torch.distributions.Categorical(action_proba)
        action = action_dist.sample()
        if self.policy.training:
            log_probas = action_dist.log_prob(action)
            self.action_logprob_history.append(log_probas)
        return action.item()

    def store_reward(self, reward):
        self.reward_history.append(reward)

    def update(self):
        # calculate MC returns & loss
        T = len(self.reward_history)
        discounts = torch.logspace(0, T, steps=T + 1, base=self.gamma, device=self.device)[:T]
        returns = torch.tensor([torch.tensor(
            self.reward_history[t:], dtype=torch.float, device=self.device) @ discounts[t:] for t in range(T)])
        loss = 0
        for g, log_prob in zip(returns, self.action_logprob_history):
            loss += - g * log_prob

        # sgd + reset history
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reward_history, self.action_logprob_history = [], []


class ActorCriticAgent:
    def __init__(self, input_dim, action_dim, hidden_dim, gamma, lr):
        self.gamma = gamma
        self.actor_critic = ActorCritic(*input_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.log_proba, self.value = None, None


    def choose_action(self, state):
        state = torch.from_numpy(state).to(self.device)
        self.value, action_logits = self.actor_critic(state)
        action_proba = torch.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_proba)
        action = action_dist.sample()
        self.log_proba = action_dist.log_prob(action)
        return action.item()

    def update(self, reward, state_, done):
        # calculate TD loss
        state_ = torch.from_numpy(state_).unsqueeze(0).to(self.device)
        value_, _ = self.actor_critic(state_)
        critic_loss = (reward + self.gamma * value_ * ~done - self.value).pow(2)

        # actor loss
        actor_loss = - self.value.detach() * self.log_proba

        # sgd + reset history
        loss = critic_loss + actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
