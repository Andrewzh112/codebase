import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from collections import deque
from pathlib import Path
from gym import wrappers
# from ple.games.flappybird import FlappyBird

from qlearning.agent import DQNAgent
from qlearning.atari.utils import processed_atari


parser = argparse.ArgumentParser(description='Q Learning Atari Agents')
parser.add_argument('--algorithm', type=str, default='DuelingDDQN', help='The type of algorithm you wish to use.\nDQN\n \
                                                                                DDQN\n \
                                                                                DuelingDQN\n \
                                                                                DuelingDDQN')

# environment & data
parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Atari environment.\nPongNoFrameskip-v4\n \
                                                                                BreakoutNoFrameskip-v4\n \
                                                                                SpaceInvadersNoFrameskip-v4\n \
                                                                                EnduroNoFrameskip-v4\n \
                                                                                AtlantisNoFrameskip-v4\n \
                                                                                BankHeistNoFrameskip-v4')
parser.add_argument('--n_repeats', type=int, default=4, help='The number of repeated actions')
parser.add_argument('--img_size', type=int, default=84, help='The height and width of images after resizing')
parser.add_argument('--input_channels', type=int, default=1, help='The input channels after preprocessing')
parser.add_argument('--hidden_dim', type=int, default=512, help='The hidden size for second fc layer')
parser.add_argument('--max_size', type=int, default=100000, help='Buffer size')
parser.add_argument('--target_update_interval', type=int, default=1000, help='Interval for updating target network')

# training
parser.add_argument('-e', '--n_episodes', '--epochs', type=int, default=1000, help='Number of episodes agent interacts with env')
parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
parser.add_argument('--epsilon_init', type=float, default=1.0, help='Initial epsilon value')
parser.add_argument('--epsilon_min', type=float, default=0.1, help='Minimum epsilon value to decay to')
parser.add_argument('--epsilon_desc', type=float, default=1e-5, help='Epsilon decrease')
parser.add_argument('--grad_clip', type=float, default=10, help='Norm of the grad clip, None for no clip')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--no_prioritize', action="store_true", default=False, help='Use Prioritized Experience Replay')
parser.add_argument('--alpha', type=float, default=0.6, help='Prioritized Experience Replay alpha')
parser.add_argument('--beta', type=float, default=0.4, help='Prioritized Experience Replay beta')
parser.add_argument('--eps', type=float, default=1e-5, help='Prioritized Experience Replay epsilon')

# logging
parser.add_argument('--progress_window', type=int, default=100, help='Window of episodes for progress')
parser.add_argument('--print_every', type=int, default=1, help='Print progress interval')
parser.add_argument('--cpt_dir', type=str, default='qlearning/atari/model_weights', help='Directory to save model weights')
parser.add_argument('--log_dir', type=str, default='qlearning/atari/logs', help='Directory to submit logs')

# testing
parser.add_argument('--test', action="store_true", default=False, help='Testing + rendering')
parser.add_argument('--video', action="store_false", default=True, help='Output video files if testing')
parser.add_argument('--video_dir', type=str, default='qlearning/atari/videos', help='Directory for agent playing videos')
args = parser.parse_args()

writer = SummaryWriter(args.log_dir)
Path(args.log_dir).mkdir(parents=True, exist_ok=True)
Path(args.cpt_dir).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    env = processed_atari(args.env_name, args.img_size, args.input_channels, args.n_repeats)

    # if testing agent and want to output videos, make dir & wrap env to auto output video files 
    if args.test and args.video:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
        env = wrappers.Monitor(env, args.video_dir,
                               video_callable=lambda episode_id: True,
                               force=True)
    if 'DQN' in args.algorithm:
        agent = DQNAgent(env.observation_space.shape,
                         env.action_space.n,
                         args.epsilon_init, args.epsilon_min, args.epsilon_desc,
                         args.gamma, args.lr, args.n_episodes,
                         input_channels=args.input_channels,
                         algorithm=args.algorithm,
                         img_size=args.img_size,
                         hidden_dim=args.hidden_dim,
                         max_size=args.max_size,
                         target_update_interval=args.target_update_interval,
                         batch_size=args.batch_size,
                         cpt_dir=args.cpt_dir,
                         grad_clip=args.grad_clip,
                         prioritize=not args.no_prioritize,
                         alpha=args.alpha,
                         beta=args.beta,
                         eps=args.eps,
                         env_name=args.env_name)
    else:
        raise NotImplementedError
    # force some parameters depending on if using priority replay
    if args.no_prioritize:
        args.alpha, args.beta, args.epsilon = 1, 0, 0
    else:
        args.lr /= 4
    scores, best_score = deque(maxlen=args.progress_window), -np.inf

    # load weights & make sure model in eval mode during test
    if args.test:
        agent.load_models()
        agent.Q_function.eval()
    pbar = tqdm(range(args.n_episodes))
    for e in pbar:

        # reset every episode and make sure functions are in training mode
        done, score, observation = False, 0, env.reset()
        agent.Q_function.train()
        while not done:
            # if test, only take greedy action, if not epsilon greedy
            if args.test:
                action = agent.greedy_action(observation)
                env.render()
            else:
                action = agent.epsilon_greedy(observation)
            next_observation, reward, done, info = env.step(action)

            # only update parameters during training
            if not args.test:
                priorities = agent.update()
                agent.store_transition(observation, reward, action, next_observation, done, priorities)
            score += reward
            observation = next_observation

        # logging
        writer.add_scalars('Performance and training', {'Score': score, 'Epsilon': agent.epsilon})
        scores.append(score)
        if score > best_score and not args.test:
            agent.save_models()
            best_score = score
        if (e + 1) % args.print_every == 0:
            tqdm.write(f'Episode: {e + 1}/{args.n_episodes}, Average Score: {np.mean(scores)}, Best Score {best_score}, Epsilon: {agent.epsilon}')
