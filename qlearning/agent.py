import pandas as pd
import numpy as np
import torch
from torch import optim
from copy import deepcopy
from qlearning.networks import QNaive, QBasic, QDueling
from qlearning.experience_replay import ReplayBuffer


class BaseAgent:
    def __init__(self, state_dim, n_actions, epsilon_init, epsilon_min, epsilon_desc, gamma, lr, n_episodes):
        self.actions = list(range(n_actions))
        self.n_actions = n_actions
        if isinstance(state_dim, int):
            self.states = list(range(state_dim))
        self.state_dim = state_dim
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_desc = epsilon_desc
        self.gamma = gamma
        self.lr = lr
        self.n_episodes = n_episodes

    def epsilon_greedy(self, state):
        if np.random.random() > self.epsilon:
            return self.greedy_action(state)
        return np.random.choice(self.actions)

    def greedy_action(self, state):
        raise NotImplementedError


class TabularAgent(BaseAgent):
    def __init__(self, states, actions, epsilon_init, epsilon_min, epsilon_desc, gamma, lr, n_episodes):
        super().__init__(states, actions, epsilon_init, epsilon_min, epsilon_desc, gamma, lr, n_episodes)
        # initialize table with 0 Q-values
        self.q_table = pd.DataFrame(np.zeros((self.state_dim, self.n_actions)),
                                    index=states, columns=actions)

    def greedy_action(self, state):
        Qs = self.q_table.loc[state]
        return Qs.argmax()

    def update(self, state, action, reward, next_state):
        # update Q-table
        max_Q_ = self.q_table.loc[next_state].max()
        Q_sa = self.q_table.loc[state, action]
        self.q_table.loc[state, action] += self.lr * (
            reward + self.gamma * max_Q_ - Q_sa)
        # update epsilon
        self.decrease_epsilon()

    def decrease_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_desc)


class NaiveNeuralAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if kwargs['policy'] == 'naive neural':
            self.Q_function = QNaive(
                kwargs['state_dim'],
                kwargs['action_dim'],
                kwargs['hidden_dim'],
                self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.Q_function.parameters(), self.lr)
        self.criterion = torch.nn.MSELoss()

    def number2tensor(self, number):
        return torch.tensor([number]).to(self.device)

    def greedy_action(self, state):
        state = self.number2tensor(state)
        next_action = self.Q_function(state).argmax()
        return next_action.item()

    def update(self, state, action, reward, next_state):
        q_prime = self.Q_function(next_state).max()
        q_target = torch.tensor([reward + self.gamma * q_prime]).to(self.device)
        q_pred = self.Q_function(state)[action]
        loss = self.criterion(q_target, q_pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decrease_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon - self.epsilon_desc)


class DQNAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.algorithm = kwargs['algorithm']
        self.batch_size = kwargs['batch_size']
        self.grad_clip = kwargs['grad_clip']
        self.prioritize = kwargs['prioritize']
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']
        self.eps = kwargs['eps']
        self.memory = ReplayBuffer(kwargs['max_size'], self.state_dim)
        self.target_update_interval = kwargs['target_update_interval']
        self.n_updates = 0

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.algorithm.startswith('Dueling'):
            self.Q_function = QDueling(
                    kwargs['input_channels'],
                    self.n_actions,
                    kwargs['cpt_dir'],
                    kwargs['algorithm'] + '_' + kwargs['env_name'],
                    kwargs['img_size'],
                    kwargs['hidden_dim']).to(self.device)
        else:
            self.Q_function = QBasic(
                    kwargs['input_channels'],
                    self.n_actions,
                    kwargs['cpt_dir'],
                    kwargs['algorithm'] + '_' + kwargs['env_name'],
                    kwargs['img_size'],
                    kwargs['hidden_dim']).to(self.device)

        # instanciate target network
        self.target_Q = deepcopy(self.Q_function)
        self.freeze_network(self.target_Q)
        self.target_Q.name = kwargs['algorithm'] + '_' + kwargs['env_name'] + '_target'

        self.optimizer = torch.optim.RMSprop(self.Q_function.parameters(), lr=self.lr, alpha=0.95)
        self.criterion = torch.nn.MSELoss(reduction='none')

    def greedy_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_action = self.Q_function(observation).argmax()
        return next_action.item()

    def update_target_network(self):
        self.copy_network_weights(self.Q_function, self.target_Q)

    def copy_network_weights(self, src_network, tgt_network):
        tgt_network.load_state_dict(src_network.state_dict())

    def freeze_network(self, network):
        for p in network.parameters():
            p.requires_grad = False

    def update(self):
        # Q_t = Q_t + lr * (reward + gamma * Q'_t - Q^target_t) ** 2
        # keep sampling until we have full batch
        if self.memory.ctr < self.batch_size:
            return
        self.optimizer.zero_grad()
        observations, rewards, actions, next_observations, dones, idx, weights = self.sample_transitions()

        # double DQN uses online network to select action for Q'
        if self.algorithm.endswith('DDQN'):
            next_actions = self.Q_function(next_observations).argmax(-1)
            q_prime = self.target_Q(next_observations)[list(range(self.batch_size)), next_actions]
        elif self.algorithm.endswith('DQN'):
            q_prime = self.target_Q(next_observations).max(-1)[0]

        # calculate target + estimate
        q_target = rewards + self.gamma * q_prime * (~dones)
        q_pred = self.Q_function(observations)[list(range(self.batch_size)), actions]
        loss = self.criterion(q_target.detach(), q_pred)

        # for updating priorities if using priority replay
        if self.prioritize:
            priorities = (idx, loss.clone().detach() + self.eps)
        else:
            priorities = None

        # update
        loss = (loss * weights).mean()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.Q_function.parameters(), self.grad_clip)
        self.optimizer.step()
        self.decrease_epsilon()
        self.n_updates += 1
        if self.n_updates % self.target_update_interval == 0:
            self.update_target_network()
        return priorities

    def decrease_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon - self.epsilon_desc)

    def store_transition(self, state, reward, action, next_state, done, priority=None):
        state, next_state = torch.from_numpy(state), torch.from_numpy(next_state)
        self.memory.store(state, reward, action, next_state, done, priority=priority)

    def sample_transitions(self):
        return self.memory.sample(self.batch_size, self.device)

    def save_models(self):
        self.target_Q.check_point()
        self.Q_function.check_point()

    def load_models(self):
        self.target_Q.load_checkpoint()
        self.Q_function.load_checkpoint()
        self.target_Q.to(self.device)
        self.Q_function.to(self.device)
