import numpy as np
import torch
from torch.nn import functional as F
from copy import deepcopy

from policy.networks import ActorCritic, Actor, Critic, SACActor, SACCritic, PPOActor, PPOCritic
from policy.utils import ReplayBuffer, EpisodeBuffer, OUActionNoise, clip_action, GaussianActionNoise


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
        TD_error = reward + self.gamma * value_ * ~done - self.value
        critic_loss = TD_error.pow(2)

        # actor loss
        actor_loss = - self.value * self.log_proba

        # sgd + reset history
        loss = critic_loss + actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dims, max_action, gamma, 
                 tau, critic_lr, critic_wd, actor_lr, actor_wd, batch_size,
                 final_init, maxsize, sigma, theta, dt, checkpoint):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.critic_lr = critic_lr
        self.critic_wd = critic_wd
        self.final_init = final_init
        self.checkpoint = checkpoint
        self.sigma = sigma

        self.memory = ReplayBuffer(state_dim, action_dim, maxsize)
        self.noise = OUActionNoise(torch.zeros(action_dim, device=self.device),
                                   sigma=sigma,
                                   theta=theta,
                                   dt=dt)
        self.critic = Critic(*state_dim, *action_dim, hidden_dims, critic_lr, critic_wd,
                             final_init, checkpoint, 'Critic')
        self.actor = Actor(*state_dim, *action_dim, hidden_dims, max_action,
                           actor_lr, actor_wd, final_init, checkpoint, 'Actor')
        self.target_critic = self.get_target_network(self.critic)
        self.target_critic.name = 'Target_Critic'
        self.target_actor = self.get_target_network(self.actor)
        self.target_actor.name = 'Target_Actor'
    
    def get_target_network(self, online_network, freeze_weights=True):
        target_network = deepcopy(online_network)
        if freeze_weights:
            for param in target_network.parameters():
                param.requires_grad = False
        return target_network

    def update(self):
        experiences = self.memory.sample_transition(self.batch_size)
        states, actions, rewards, next_states, dones = [data.to(self.device) for data in experiences]

        # actor loss is by maximizing Q values
        self.actor.optimizer.zero_grad()
        qs = self.critic(states, self.actor(states))
        actor_loss = - qs.mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        # calculate targets & only update online critic network
        self.critic.optimizer.zero_grad()
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            q_primes = self.target_critic(next_states, target_actions).squeeze()
            targets = rewards + self.gamma * q_primes * (~dones)
        qs = self.critic(states, actions)
        critic_loss = F.mse_loss(targets.unsqueeze(-1), qs)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.update_target_network(self.critic, self.target_critic)
        self.update_target_network(self.actor, self.target_actor)

        return actor_loss.item(), critic_loss.item()

    def update_target_network(self, src, tgt):
        for src_weight, tgt_weight in zip(src.parameters(), tgt.parameters()):
            tgt_weight.data = tgt_weight.data * self.tau + src_weight.data * (1. - self.tau)

    def save_models(self, info):
        self.critic.save_checkpoint(info)
        self.actor.save_checkpoint(info)
        self.target_critic.save_checkpoint(info)
        self.target_actor.save_checkpoint(info)

    def load_models(self, info):
        self.critic.load_checkpoint(info)
        self.actor.load_checkpoint(info)
        self.target_critic.load_checkpoint(info)
        self.target_actor.load_checkpoint(info)

    def choose_action(self, observation, test):
        self.actor.eval()
        observation = torch.from_numpy(observation).to(self.device)
        with torch.no_grad():
            mu = self.actor(observation)
        if test:
            action = mu
        else:
            action = mu + self.noise()
        self.actor.train()
        action = action.cpu().detach().numpy()
        # clip noised action to ensure not out of bounds
        return clip_action(action, self.max_action)

    def store_transition(self, state, action, reward, next_state, done):
        state = torch.tensor(state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_state = torch.tensor(next_state)
        done = torch.tensor(done, dtype=torch.bool)
        self.memory.store_transition(state, action, reward, next_state, done)


class TD3Agent(DDPGAgent):
    def __init__(self, *args, **kwargs):
        exluded_kwargs = ['actor_update_iter', 'action_sigma', 'action_clip']
        super().__init__(*args, **{k: v for k, v in kwargs.items() if k not in exluded_kwargs})
        self.ctr = 0
        self.actor_update_iter = kwargs['actor_update_iter']
        self.action_sigma = kwargs['action_sigma']
        self.action_clip = kwargs['action_clip']
        self.noise = GaussianActionNoise(mu=0, sigma=self.sigma)
        self.actor_loss = 0

        # second pair of critic
        self.critic2 = Critic(*self.state_dim, *self.action_dim, self.hidden_dims,
                              self.critic_lr, self.critic_wd,
                              self.final_init, self.checkpoint, 'Critic2')
        self.target_critic2 = self.get_target_network(self.critic2)
        self.target_critic2.name = 'Target_Critic2'

    def choose_action(self, observation, test):
        self.actor.eval()
        self.ctr += 1
        observation = torch.from_numpy(observation).to(self.device)
        with torch.no_grad():
            action = self.actor(observation)
        if not test:
            action = action + self.noise(action.size()).to(self.device)
        self.actor.train()
        action = action.cpu().detach().numpy()
        # clip noised action to ensure not out of bounds
        return clip_action(action, self.max_action)

    def update(self):
        experiences = self.memory.sample_transition(self.batch_size)
        states, actions, rewards, next_states, dones = [data.to(self.device) for data in experiences]

        # actor loss is by maximizing Q values
        if self.ctr % self.actor_update_iter == 0:
            self.actor.optimizer.zero_grad()
            qs = self.critic(states, self.actor(states))
            actor_loss = - qs.mean()
            actor_loss.backward()
            self.actor.optimizer.step()
            self.actor_loss = actor_loss.item()

            self.update_target_network(self.critic, self.target_critic)
            self.update_target_network(self.critic2, self.target_critic2)
            self.update_target_network(self.actor, self.target_actor)

        # calculate targets & only update online critic network
        self.critic.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        with torch.no_grad():
            # y <- r + gamma * min_(i=1,2) Q_(theta'_i)(s', a_telda)
            target_actions = self.target_actor(next_states)
            target_actions += self.noise(
                target_actions.size(), clip=self.action_clip, sigma=self.action_sigma).to(self.device)
            target_actions = clip_action(target_actions.cpu().numpy(), self.max_action)
            target_actions = torch.from_numpy(target_actions).to(self.device)
            q_primes1 = self.target_critic(next_states, target_actions).squeeze()
            q_primes2 = self.target_critic2(next_states, target_actions).squeeze()
            q_primes = torch.min(q_primes1, q_primes2)
            targets = rewards + self.gamma * q_primes * (~dones)
        # theta_i <- argmin_(theta_i) N^(-1) sum(y - Q_(theta_i)(s, a))^2
        qs1 = self.critic(states, actions)
        qs2 = self.critic2(states, actions)
        critic_loss1 = F.mse_loss(targets.unsqueeze(-1), qs1)
        critic_loss2 = F.mse_loss(targets.unsqueeze(-1), qs2)
        critic_loss = critic_loss1 + critic_loss2
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic2.optimizer.step()
        return self.actor_loss, critic_loss.item()


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dims, max_action, gamma, 
                 tau, alpha, lr, batch_size, maxsize, log_std_min, log_std_max,
                 epsilon, checkpoint):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha

        self.memory = ReplayBuffer(state_dim, action_dim, maxsize)
        self.critic1 = SACCritic(*state_dim, *action_dim, hidden_dims, lr,
                                 checkpoint, 'Critic')
        self.critic2 = SACCritic(*state_dim, *action_dim, hidden_dims,
                                 lr, checkpoint, 'Critic2')
        self.actor = SACActor(*state_dim, *action_dim, hidden_dims, max_action,
                              log_std_min, log_std_max, epsilon, lr,
                              checkpoint, 'Actor')
        self.target_critic1 = self.get_target_network(self.critic1)
        self.target_critic1.name = 'Target_Critic1'
        self.target_critic2 = self.get_target_network(self.critic2)
        self.target_critic2.name = 'Target_Critic2'

    def get_target_network(self, online_network, freeze_weights=True):
        target_network = deepcopy(online_network)
        if freeze_weights:
            for param in target_network.parameters():
                param.requires_grad = False
        return target_network

    def choose_action(self, observation, test):
        self.actor.eval()
        observation = torch.from_numpy(observation).to(self.device)
        with torch.no_grad():
            action, _ = self.actor(observation)
        self.actor.train()
        action = action.cpu().detach().numpy()
        return action

    def update(self):
        experiences = self.memory.sample_transition(self.batch_size)
        states, actions, rewards, next_states, dones = [data.to(self.device) for data in experiences]

        ###### UPDATE CRITIC ######
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            next_q1 = self.target_critic1(next_states, next_actions).squeeze()
            next_q2 = self.target_critic2(next_states, next_actions).squeeze()
            action_values = torch.min(next_q1, next_q2) - self.alpha * next_log_probs.squeeze()
            targets = rewards + self.gamma * action_values * (~dones)

        q1 = self.critic1(states, actions).squeeze()
        q2 = self.critic2(states, actions).squeeze()
        critic_loss1 = 0.5 * F.mse_loss(targets, q1)
        critic_loss2 = 0.5 * F.mse_loss(targets, q2)
        critic_loss = critic_loss1 + critic_loss2
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        ###### UPDATE ACTOR ######
        self.actor.optimizer.zero_grad()
        actions, log_probs = self.actor(states)
        q1 = self.critic1(states, actions).squeeze()
        q2 = self.critic2(states, actions).squeeze()
        q = torch.min(q1, q2)
        actor_loss = torch.mean(self.alpha * log_probs.squeeze() - q)
        actor_loss.backward()
        self.actor.optimizer.step()

        ###### UPDATE TARGET CRITICS ######
        self.update_target_network(self.critic1, self.target_critic1)
        self.update_target_network(self.critic2, self.target_critic2)

        return critic_loss.item(), actor_loss.item()

    def update_target_network(self, src, tgt):
        for src_weight, tgt_weight in zip(src.parameters(), tgt.parameters()):
            tgt_weight.data = tgt_weight.data * self.tau + src_weight.data * (1. - self.tau)

    def store_transition(self, state, action, reward, next_state, done):
        state = torch.tensor(state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_state = torch.tensor(next_state)
        done = torch.tensor(done, dtype=torch.bool)
        self.memory.store_transition(state, action, reward, next_state, done)

    def save_models(self, info):
        self.critic1.save_checkpoint(info)
        self.critic2.save_checkpoint(info)
        self.actor.save_checkpoint(info)
        self.target_critic1.save_checkpoint(info)
        self.target_critic2.save_checkpoint(info)

    def load_models(self, info):
        self.critic1.load_checkpoint(info)
        self.critic2.load_checkpoint(info)
        self.actor.load_checkpoint(info)
        self.target_critic1.load_checkpoint(info)
        self.target_critic2.load_checkpoint(info)


class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dims, max_action,
                 log_std_min, log_std_max, epsilon, gamma, lambda_, lr, clip,
                 batch_size, H, n_epochs, checkpoint):
        self.gamma = gamma
        self.clip = clip
        self.H = H
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.actor = PPOActor(*state_dim, *action_dim, hidden_dims, max_action,
                              log_std_min, log_std_max, epsilon, lr,
                              checkpoint, 'Actor')
        self.critic = PPOCritic(*state_dim, hidden_dims, lr,
                                checkpoint, 'Critic')
        self.memory = EpisodeBuffer(self.device)

    def store_transition(self, states, actions, rewards, values, log_probs, dones):
        self.memory.store_transition(states, actions, rewards, values, log_probs, dones)

    def save_models(self, info):
        self.critic.save_checkpoint(info)
        self.actor.save_checkpoint(info)

    def load_models(self, info):
        self.critic.load_checkpoint(info)
        self.actor.load_checkpoint(info)

    def choose_action(self, state):
        state = torch.tensor(state).to(self.device)
        with torch.no_grad():
            action, log_probs = self.actor(state)
            value = self.critic(state)
        return action.squeeze().item(), log_probs.squeeze().item(), value.squeeze().item()

    def update(self):
        for _ in range(self.n_epochs):
            states, actions, rewards, values, log_probs, dones = self.memory.sample_batch(self.batch_size)
