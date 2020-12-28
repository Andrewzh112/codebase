import os
from itertools import chain

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from td3.utils import ReplayBuffer
from td3.networks import Actor, Critic, ImageActor, ImageCritic


class Agent:
    def __init__(self, env, alpha, beta, hidden_dims, tau,
                 batch_size, gamma, d, warmup, max_size, c,
                 sigma, one_device, log_dir, checkpoint_dir,
                 img_input, in_channels, order, depth, multiplier,
                 action_embed_dim, hidden_dim, crop_dim):
        if img_input:
            input_dim = [in_channels*order, crop_dim, crop_dim]
        else:
            input_dim = env.observation_space.shape
            state_space = input_dim[0]
        n_actions = env.action_space.shape[0]

        # training params
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.buffer = ReplayBuffer(max_size, input_dim, n_actions)
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.d = d
        self.c = c
        self.sigma = sigma
        self.img_input = img_input

        # training device
        if one_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # logging/checkpointing
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = checkpoint_dir

        # networks & optimizers
        if img_input:
            self.actor = ImageActor(in_channels, n_actions, hidden_dim, self.max_action, order, depth, multiplier, 'actor').to(self.device)
            self.critic_1 = ImageCritic(in_channels, n_actions, hidden_dim, action_embed_dim, order, depth, multiplier, 'critic_1').to(self.device)
            self.critic_2 = ImageCritic(in_channels, n_actions, hidden_dim, action_embed_dim, order, depth, multiplier, 'critic_2').to(self.device)

            self.target_actor = ImageActor(in_channels, n_actions, hidden_dim, self.max_action, order, depth, multiplier, 'target_actor').to(self.device)
            self.target_critic_1 = ImageCritic(in_channels, n_actions, hidden_dim, action_embed_dim, order, depth, multiplier, 'target_critic_1').to(self.device)
            self.target_critic_2 = ImageCritic(in_channels, n_actions, hidden_dim, action_embed_dim, order, depth, multiplier, 'target_critic_2').to(self.device)
            print('actor')
            print(self.actor)
            
        # physics networks
        else:
            self.actor = Actor(state_space, hidden_dims, n_actions, self.max_action, 'actor').to(self.device)
            self.critic_1 = Critic(state_space, hidden_dims, n_actions, 'critic_1').to(self.device)
            self.critic_2 = Critic(state_space, hidden_dims, n_actions, 'critic_2').to(self.device)

            self.target_actor = Actor(state_space, hidden_dims, n_actions, self.max_action, 'target_actor').to(self.device)
            self.target_critic_1 = Critic(state_space, hidden_dims, n_actions, 'target_critic_1').to(self.device)
            self.target_critic_2 = Critic(state_space, hidden_dims, n_actions, 'target_critic_2').to(self.device)

        self.critic_optimizer = torch.optim.Adam(
                chain(self.critic_1.parameters(), self.critic_2.parameters()), lr=beta)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)

        # copy weights
        self.update_network_parameters(tau=1)

    def _get_noise(self, clip=True):
        noise = torch.randn(self.n_actions, dtype=torch.float, device=self.device) * self.sigma
        if clip:
            noise = noise.clamp(-self.c, self.c)
        return noise

    def _clamp_action_bound(self, action):
        return action.clamp(self.min_action, self.max_action)

    def choose_action(self, observation, rendering=False):
        if self.time_step < self.warmup or not rendering:
            mu = self._get_noise(clip=False)
        else:
            state = torch.tensor(observation, dtype=torch.float).to(self.device)
            mu = self.actor(state) + self._get_noise(clip=False)
        self.time_step += 1
        return self._clamp_action_bound(mu).cpu().detach().numpy()

    def remember(self, state, action, reward, state_, done):
        self.buffer.store_transition(state, action, reward, state_, done)

    def critic_step(self, state, action, reward, state_, done):
        # get target actions w/ noise
        target_actions = self.target_actor(state_) + self._get_noise()
        target_actions = self._clamp_action_bound(target_actions)

        # target & online values
        q1_ = self.target_critic_1(state_, target_actions)
        q2_ = self.target_critic_2(state_, target_actions)

        # done mask
        q1_[done], q2_[done] = 0.0, 0.0

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)        

        critic_value_ = torch.min(q1_, q2_)

        target = reward + self.gamma * critic_value_
        target = target.unsqueeze(1)

        self.critic_optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_optimizer.step()

        self.writer.add_scalar('Critic loss', critic_loss.item(), global_step=self.learn_step_counter)

    def actor_step(self, state):
        # calculate loss, update actor params
        self.actor_optimizer.zero_grad()
        actor_loss = -torch.mean(self.critic_1(state, self.actor(state)))
        actor_loss.backward()
        self.actor_optimizer.step()

        # update & log
        self.update_network_parameters()
        self.writer.add_scalar('Actor loss', actor_loss.item(), global_step=self.learn_step_counter)

    def learn(self):
        self.learn_step_counter += 1

        # if the buffer is not yet filled w/ enough samples
        if self.buffer.counter < self.batch_size:
            return

        # transitions
        state, action, reward, state_, done = self.buffer.sample_buffer(self.batch_size)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        done = torch.tensor(done).to(self.device)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state_ = torch.tensor(state_, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)

        self.critic_step(state, action, reward, state_, done)
        if self.learn_step_counter % self.d == 0:
            self.actor_step(state)

    def momentum_update(self, online_network, target_network, tau):
        for param_o, param_t in zip(online_network.parameters(), target_network.parameters()):
            param_t.data = param_t.data * tau + param_o.data * (1. - tau)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        self.momentum_update(self.critic_1, self.target_critic_1, tau)
        self.momentum_update(self.critic_2, self.target_critic_2, tau)
        self.momentum_update(self.actor, self.target_actor, tau)

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step=global_step)

    def save_networks(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
        }, self.checkpoint_dir)

    def load_state_dicts(self):
        state_dict = torch.load(self.checkpoint_dir)
        self.actor.load_state_dict(state_dict['actor'])
        self.target_actor.load_state_dict(state_dict['target_actor'])
        self.critic_1.load_state_dict(state_dict['critic_1'])
        self.critic_2.load_state_dict(state_dict['critic_2'])
        self.target_critic_1.load_state_dict(state_dict['target_critic_1'])
        self.target_critic_2.load_state_dict(state_dict['target_critic_2'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
