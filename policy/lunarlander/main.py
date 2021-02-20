import gym
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import deque
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from policy import agent as Agent


parser = argparse.ArgumentParser(description='Lunar Lander Agents')
# training hyperparams
parser.add_argument('--agent', type=str, default='DDPG', help='Agent style')
parser.add_argument('--n_episodes', type=int, default=3000, help='Number of episodes you wish to run for')
parser.add_argument('--batch_size', type=int, default=64, help='Minibatch size')
parser.add_argument('--hidden_dim', type=int, default=2048, help='Hidden dimension of FC layers')
parser.add_argument('--hidden_dims', type=list, default=[400, 300], help='Hidden dimensions of FC layers')
parser.add_argument('--critic_lr', type=float, default=1e-3, help='Learning rate for Critic')
parser.add_argument('--critic_wd', type=float, default=1e-2, help='Weight decay for Critic')
parser.add_argument('--actor_lr', type=float, default=1e-4, help='Learning rate for Actor')
parser.add_argument('--actor_wd', type=float, default=0., help='Weight decay for Actor')
parser.add_argument('--gamma', type=float, default=0.99, help='Reward discount factor')
parser.add_argument('--final_init', type=float, default=3e-3, help='The range for output layer initialization')
parser.add_argument('--tau', type=float, default=0.001, help='Weight of target network update')
parser.add_argument('--maxsize', type=int, default=1e6, help='Size of Replay Buffer')
parser.add_argument('--sigma', type=float, default=0.2, help='Sigma for UOnoise')
parser.add_argument('--theta', type=float, default=0.15, help='Theta for UOnoise')
parser.add_argument('--dt', type=float, default=1e-2, help='dt for UOnoise')

# eval params
parser.add_argument('--render', action="store_true", default=False, help='Render environment while training')
parser.add_argument('--window_legnth', type=int, default=100, help='Length of window to keep track scores')

# checkpoint + logs
parser.add_argument('--checkpoint', type=str, default='policy/lunarlander/checkpoint', help='Checkpoint for model weights')
parser.add_argument('--logdir', type=str, default='policy/lunarlander/logs', help='Directory to save logs')
args = parser.parse_args()


def main():
    env_type = 'Continuous' if args.agent in ['DDPG'] else ''
    env = gym.make(f'LunarLander{env_type}-v2')
    agent_ = getattr(Agent, args.agent.replace(' ', '') + 'Agent')
    if args.agent in ['DDPG']:
        max_action = float(env.action_space.high[0])
        agent = agent_(state_dim=env.observation_space.shape,
                       action_dim=env.action_space.shape,
                       hidden_dims=args.hidden_dims,
                       max_action=max_action,
                       gamma=args.gamma,
                       tau=args.tau,
                       critic_lr=args.critic_lr,
                       critic_wd=args.critic_wd,
                       actor_lr=args.actor_lr,
                       actor_wd=args.actor_wd,
                       batch_size=args.batch_size,
                       final_init=args.final_init,
                       maxsize=int(args.maxsize),
                       sigma=args.sigma,
                       theta=args.theta,
                       dt=args.dt,
                       checkpoint=args.checkpoint)
    else:
        agent = agent_(state_dim=env.observation_space.shape,
                       actionaction_dim_dim=env.action_space.n,
                       hidden_dims=args.hidden_dims,
                       gamma=args.gamma,
                       lr=args.lr)

    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(args.logdir)

    pbar = tqdm(range(args.n_episodes))
    score_history = deque(maxlen=args.window_legnth)
    best_score = env.reward_range[0]
    for e in pbar:
        done, score, observation = False, 0, env.reset()

        # reset DDPG UO Noise and also keep track of actor/critic losses
        if args.agent in ['DDPG']:
            agent.noise.reset()
            actor_losses, critic_losses = [], []
        while not done:
            if args.render:
                env.render()

            action = agent.choose_action(observation)
            next_observation, reward, done, _ = env.step(action)
            score += reward

            # update for td methods, recording for mc methods
            if args.agent == 'Actor Critic':
                agent.update(reward, next_observation, done)
            elif args.agent in ['DDPG']:
                agent.store_transition(observation, action, reward, next_observation, done)
                # if we have memory smaller than batch size, do not update
                if agent.memory.idx < args.batch_size:
                    continue
                actor_loss, critic_loss = agent.update()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                pbar.set_postfix({'Reward': reward, 'Actor Loss': actor_loss, 'Critic Loss': critic_loss})
            else:
                agent.store_reward(reward)
            observation = next_observation

        score_history.append(score)

        # update for mc methods w/ full trajectory
        if args.agent == 'Policy Gradient':
            agent.update()

        # logging & saving
        elif args.agent in ['DDPG']:
            writer.add_scalars(
                'Scores',
                {'Episodic': score, 'Windowed Average': np.mean(score_history)},
                global_step=e)
            if actor_losses:
                writer.add_scalars(
                    'Losses',
                    {'Actor': np.mean(actor_losses), 'Critic': np.mean(critic_losses)},
                    global_step=e)
            actor_losses, critic_losses = [], []

            if np.mean(score_history) > best_score:
                best_score = np.mean(score_history)
                agent.save_models()
        tqdm.write(
            f'Episode: {e + 1}/{args.n_episodes}, Score: {score}, Average Score: {np.mean(score_history)}')


if __name__ == '__main__':
    main()
