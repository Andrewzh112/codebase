import pybullet_envs
import gym
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import deque
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from policy import agent as Agent


parser = argparse.ArgumentParser(description='Continuous Environment Agents')
# training hyperparams
parser.add_argument('--agent', type=str, default='SAC', help='Agent Algorithm')
parser.add_argument('--environment', type=str, default='LunarLanderContinuous-v2', help='Agent Algorithm')
parser.add_argument('--n_episodes', type=int, default=3000, help='Number of episodes you wish to run for')
parser.add_argument('--batch_size', type=int, default=256, help='Minibatch size')
parser.add_argument('--hidden_dim', type=int, default=2048, help='Hidden dimension of FC layers')
parser.add_argument('--hidden_dims', type=list, default=[400, 300], help='Hidden dimensions of FC layers')
parser.add_argument('--critic_lr', type=float, default=1e-3, help='Learning rate for Critic')
parser.add_argument('--critic_wd', type=float, default=0., help='Weight decay for Critic')
parser.add_argument('--actor_lr', type=float, default=1e-4, help='Learning rate for Actor')
parser.add_argument('--actor_wd', type=float, default=0., help='Weight decay for Actor')
parser.add_argument('--gamma', type=float, default=0.99, help='Reward discount factor')
parser.add_argument('--final_init', type=float, default=3e-3, help='The range for output layer initialization')
parser.add_argument('--tau', type=float, default=0.005, help='Weight of target network update')
parser.add_argument('--maxsize', type=int, default=1e6, help='Size of Replay Buffer')
parser.add_argument('--sigma', type=float, default=0.1, help='Sigma for Noise')
parser.add_argument('--theta', type=float, default=0.15, help='Theta for UOnoise')
parser.add_argument('--dt', type=float, default=1e-2, help='dt for UOnoise')
parser.add_argument('--warmup_steps', type=int, default=10000, help='Warmup steps to take random actions before updating')
parser.add_argument('--actor_update_iter', type=int, default=2, help='Update actor and target network every')
parser.add_argument('--action_sigma', type=float, default=0.2, help='Std of noise for actions')
parser.add_argument('--action_clip', type=float, default=0.5, help='Max action bound')
parser.add_argument('--alpha', type=float, default=0.2, help='Entropy coeff for Soft Actor-Critic')
parser.add_argument('--log_std_min', type=float, default=-20, help='Min log std for Soft Actor')
parser.add_argument('--log_std_max', type=float, default=2, help='Max log std for Soft Actor')
parser.add_argument('--epsilon', type=float, default=1e-6, help='Prevent div(0)')

# eval params
parser.add_argument('--render', action="store_true", default=False, help='Render environment while training')
parser.add_argument('--window_legnth', type=int, default=100, help='Length of window to keep track scores')
parser.add_argument('--test', action="store_true", default=False, help='Whether to test environment')
parser.add_argument('--load_models', action="store_true", default=False, help='Load pretrained models')

# checkpoint + logs
parser.add_argument('--checkpoint', type=str, default='policy/continuous/checkpoint', help='Checkpoint for model weights')
parser.add_argument('--logdir', type=str, default='policy/continuous/logs', help='Directory to save logs')
args = parser.parse_args()


def main():
    env = gym.make(args.environment)
    agent_ = getattr(Agent, args.agent.replace(' ', '') + 'Agent')
    if args.test:
        args.load_models = True
        args.render = True
    print(args)
    if args.agent == 'DDPG':
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
    elif args.agent == 'TD3':
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
                       checkpoint=args.checkpoint,
                       actor_update_iter=args.actor_update_iter,
                       action_sigma=args.action_sigma,
                       action_clip=args.action_clip
                       )
    elif args.agent == 'SAC':
        max_action = float(env.action_space.high[0])
        agent = agent_(state_dim=env.observation_space.shape,
                       action_dim=env.action_space.shape,
                       hidden_dims=args.hidden_dims,
                       max_action=max_action,
                       gamma=args.gamma,
                       tau=args.tau,
                       alpha=args.alpha,
                       lr=args.critic_lr,
                       batch_size=args.batch_size,
                       maxsize=int(args.maxsize),
                       log_std_min=args.log_std_min,
                       log_std_max=args.log_std_max,
                       epsilon=args.epsilon,
                       checkpoint=args.checkpoint,
                       )

    else:
        agent = agent_(state_dim=env.observation_space.shape,
                       actionaction_dim_dim=env.action_space.n,
                       hidden_dims=args.hidden_dims,
                       gamma=args.gamma,
                       lr=args.lr)

    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(args.logdir)

    if args.load_models:
        agent.load_models(args.agent + '_' + args.environment)
    pbar = tqdm(range(args.n_episodes))
    score_history = deque(maxlen=args.window_legnth)
    best_score = - np.inf
    for e in pbar:
        done, score, observation = False, 0, env.reset()

        # reset DDPG UO Noise and also keep track of actor/critic losses
        if args.agent in ['DDPG', 'TD3', 'SAC']:
            if args.agent == 'DDPG':
                agent.noise.reset()
            actor_losses, critic_losses = [], []
        while not done:
            if args.render:
                env.render(mode='human')

            action = agent.choose_action(observation, args.test)
            next_observation, reward, done, _ = env.step(action)
            score += reward

            # update for td methods, recording for mc methods
            if args.test:
                continue
            elif args.agent == 'Actor Critic':
                agent.update(reward, next_observation, done)
            elif args.agent in ['DDPG', 'TD3', 'SAC']:
                agent.store_transition(observation, action, reward, next_observation, done)
                # if we have memory smaller than batch size, do not update
                if agent.memory.idx < args.batch_size or (args.agent == 'TD3' and agent.ctr < args.warmup_steps):
                    continue
                else:
                    actor_loss, critic_loss = agent.update()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                pbar.set_postfix({'Reward': reward, 'Actor Loss': actor_loss, 'Critic Loss': critic_loss})
            else:
                agent.store_reward(reward)
            observation = next_observation

        score_history.append(score)

        if args.test:
            continue
        # update for mc methods w/ full trajectory
        elif args.agent == 'Policy Gradient':
            agent.update()

        # logging & saving
        elif args.agent in ['DDPG', 'TD3', 'SAC']:
            writer.add_scalars(
                'Scores',
                {'Episodic': score, 'Windowed Average': np.mean(score_history)},
                global_step=e)

            if actor_losses:
                loss_dict = {'Actor': np.mean(actor_losses), 'Critic': np.mean(critic_losses)}
                writer.add_scalars(
                    'Losses',
                    loss_dict,
                    global_step=e)
            actor_losses, critic_losses = [], []

            if np.mean(score_history) > best_score:
                best_score = np.mean(score_history)
                agent.save_models(args.agent + '_' + args.environment)

        tqdm.write(
            f'Episode: {e + 1}/{args.n_episodes}, Score: {score}, Average Score: {np.mean(score_history)}')


if __name__ == '__main__':
    main()
