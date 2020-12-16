"""
https://www.youtube.com/watch?v=ZhFO8EWADmY&ab_channel=MachineLearningwithPhil
"""

import argparse
import gym
from tqdm import tqdm
from pathlib import Path
from collections import deque

from td3.agent import Agent


parser = argparse.ArgumentParser()

# agent hyperparameters
parser.add_argument('--env_name', type=str, default='Pendulum-v0', help='Gyme env name')
parser.add_argument('--hidden_dims', type=list, default=[400, 300], help='List of hidden dims for fc network')
parser.add_argument('--tau', type=float, default=0.005, help='Soft update param')
parser.add_argument('--gamma', type=float, default=0.99, help='Reward discount factor')
parser.add_argument('--sigma', type=float, default=0.2, help='Gaussian noise standard deviation')
parser.add_argument('--c', type=float, default=0.5, help='Noise clip')

# training hp params
parser.add_argument('--n_episodes', type=int, default=1000, help='Number of episodes')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--alpha', type=float, default=0.001, help='Learning rate actor')
parser.add_argument('--beta', type=float, default=0.001, help='Learning rate critic')
parser.add_argument('--warmup', type=int, default=1000, help='Number of warmup steps')
parser.add_argument('--d', type=int, default=2, help='Skip iteration')
parser.add_argument('--max_size', type=int, default=1000000, help='Replay buffer size')
parser.add_argument('--no_render', action="store_true", default=False, help='Whether to render')
parser.add_argument('--window_size', type=int, default=100, help='Score tracking moving average window size')

# misc
parser.add_argument('--one_device', action="store_false", default=True, help='Whether to only train on device 0')
parser.add_argument('--log_dir', type=str, default='td3/logs', help='Path to where log files will be saved')
parser.add_argument('--checkpoint_dir', type=str, default='td3/network_weights', help='Path to where model weights will be saved')
args = parser.parse_args()


if __name__ == '__main__':
    # paths
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir += f'/{args.env_name}_td3.pth'

    # env & agent
    env = gym.make(args.env_name)
    agent = Agent(env, args.alpha, args.beta, args.hidden_dims, args.tau, args.batch_size,
                  args.gamma, args.d, args.warmup, args.max_size, args.c, args.sigma,
                  args.one_device, args.log_dir, args.checkpoint_dir)

    best_score = env.reward_range[0]
    score_history = deque([], maxlen=args.window_size)
    episodes = tqdm(range(args.n_episodes))

    for e in episodes:
        # resetting
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, state_, done)
            agent.learn()

            # reset, log & render
            score += reward
            state = state_
            episodes.set_postfix({'Reward': reward})
            if args.no_render:
                continue
            env.render()

        # logging
        score_history.append(score)
        moving_avg = sum(score_history) / len(score_history)
        agent.add_scalar('Average Score', moving_avg, global_step=e)
        tqdm.write(f'Episode: {e + 1}/{args.n_episodes}, \
                Episode Score: {score}, \
                Average Score: {moving_avg}, \
                Best Score: {best_score}')

        # save weights @ best score
        if moving_avg > best_score:
            best_score = moving_avg
            agent.save_networks()
