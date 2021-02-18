import gym
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import deque
from policy import agent as Agent


parser = argparse.ArgumentParser(description='Lunar Lander Agents')
parser.add_argument('--agent', type=str, default='Actor Critic', help='Agent style')
parser.add_argument('--n_episodes', type=int, default=3000, help='Number of episodes you wish to run for')
parser.add_argument('--hidden_dim', type=int, default=2048, help='Hidden dimension of FC layers')
parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help='Learning rate for Adam optimizer')
parser.add_argument('--gamma', type=float, default=0.99, help='Reward discount factor')

parser.add_argument('--render', action="store_true", default=False, help='Render environment while training')
parser.add_argument('--window_legnth', type=int, default=100, help='Length of window to keep track scores')
args = parser.parse_args()


def main():
    env = gym.make('LunarLander-v2')
    agent_ = getattr(Agent, args.agent.replace(' ', '') + 'Agent')
    agent = agent_(input_dim=env.observation_space.shape,
                   action_dim=env.action_space.n,
                   hidden_dim=args.hidden_dim,
                   gamma=args.gamma,
                   lr=args.lr)
    pbar = tqdm(range(args.n_episodes))
    score_history = deque(maxlen=args.window_legnth)
    for e in pbar:
        done, score, observation = False, 0, env.reset()
        while not done:
            if args.render:
                env.render()
            action = agent.choose_action(observation)
            next_observation, reward, done, _ = env.step(action)
            if args.agent == 'Actor Critic':
                agent.update(reward, next_observation, done)
            else:
                agent.store_reward(reward)
            observation = next_observation
            score += reward
        if args.agent == 'Policy Gradient':
            agent.update()
        score_history.append(score)
        tqdm.write(
            f'Episode: {e + 1}/{args.n_episodes}, Score: {score}, Average Score: {np.mean(score_history)}')


if __name__ == '__main__':
    main()
