import matplotlib.pyplot as plt
import gym
import numpy as np
import argparse
from collections import deque
import torch
from tqdm import tqdm
from qlearning.agents import TabularAgent, NaiveNeuralAgent


parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='naive neural', help='The type of policy you wish to use')
parser.add_argument('--trailing_n', type=int, default=10, help='Window size of plotting win %')
parser.add_argument('--n_episodes', type=int, default=10000, help='Number of episodes agent interacts with env')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
parser.add_argument('--epsilon_init', type=float, default=1.0, help='Initial epsilon value')
parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum epsilon value to decay to')
parser.add_argument('--epsilon_desc', type=float, default=0.000396, help='Epsilon decrease')
parser.add_argument('--progress_window', type=int, default=100, help='Window of episodes for progress')
parser.add_argument('--print_every', type=int, default=1000, help='Print progress interval')
args = parser.parse_args()


class Policies:
    """
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    """
    def __init__(self, policy):
        self.policy = policy
        if policy == 'tabular':
            self.agent = TabularAgent(np.arange(env.observation_space.n),
                                       np.arange(env.action_space.n),
                                       args.epsilon_init, args.epsilon_min, args.epsilon_desc,
                                       args.gamma, args.lr, args.n_episodes)
        elif policy == 'naive neural':
            self.agent = NaiveNeuralAgent(np.arange(env.observation_space.n),
                                       np.arange(env.action_space.n),
                                       args.epsilon_init, args.epsilon_min, args.epsilon_desc,
                                       args.gamma, args.lr, args.n_episodes,
                                       policy=policy, state_dim=64, action_dim=64, hidden_dim=128)

    def __call__(self, state):
        if self.policy == 'random':
            return env.action_space.sample()
        if self.policy == 'direct':
            return self._direct_policy(state)
        if self.policy in ['tabular', 'naive neural']:
            return self._epsilon_greedy(state)

    def _direct_policy(self, state):
        if state in [0, 4, 6, 9, 10]:
            return 1
        if state in [1, 2, 8, 13, 14]:
            return 2
        if state in [3]:
            return 0

    def _epsilon_greedy(self, state):
        return self.agent.epsilon_greedy(state)

    def update(self, state, action, reward, next_state):
        self.agent.update(state, action, reward, next_state)


def plot_win_perc(scores, trailing_n, n_episodes):
    win_perc = [sum(scores[i - trailing_n:i]) / trailing_n for i in range(trailing_n, n_episodes)]
    plt.figure(figsize=(12, 12))
    plt.plot(win_perc)
    plt.xlabel('Number of Trials')
    plt.ylabel('Winning Percentage')
    plt.title(f'Win % over Trialing {trailing_n} Games')
    plt.show()


def plot_avg_score(scores):
    plt.figure(figsize=(12, 12))
    plt.plot(scores)
    plt.ylabel('Score')
    plt.title(f'Average Score of Last {args.progress_window} Games')
    plt.show()


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    scores, avg_scores = [], []
    pi = Policies(args.policy)
    if args.policy == 'naive neural':
        device = pi.agent.device
    pbar = tqdm(range(args.n_episodes))
    for i in pbar:
        done, observation, score = False, env.reset(), 0
        while not done:
            if args.policy == 'naive neural':
                observation = torch.tensor([observation]).to(device)
            action = pi(observation)
            next_observation, reward, done, info = env.step(action)
            if args.policy == 'naive neural':
                action, next_observation = (
                    torch.tensor([action]).to(device),
                    torch.tensor([next_observation]).to(device))
            if args.policy in ['tabular', 'naive neural']:
                pi.update(observation, action, reward, next_observation)
            score += reward
            observation = next_observation
        if args.policy == 'naive neural':
            pi.agent.decrease_epsilon()
        scores.append(score)
        avg_scores.append(np.mean(scores[-100:]))
        if (i + 1) % args.print_every == 0 and args.policy in ['tabular', 'naive neural']:
            tqdm.write(f'Episode: {i + 1}/{args.n_episodes}, Average Score: {avg_scores[-1]}, Epsilon: {pi.agent.epsilon}')
    env.close()

    # plotting
    if args.policy in ['tabular', 'naive neural']:
        plot_avg_score(avg_scores)
    else:
        plot_win_perc(scores, args.trailing_n, args.n_episodes)
