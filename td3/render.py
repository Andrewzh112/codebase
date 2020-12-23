import gym
from tqdm import tqdm
import torch
from collections import deque

from td3.agent import Agent
from td3.main import args
from networks.utils import load_weights


if __name__ == '__main__':
    args.checkpoint_dir += f'/{args.env_name}_td3.pth'
    # env & agent
    env = gym.make(args.env_name)

    if args.virtual_display:
        import pyvirtualdisplay
        _display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
        _ = _display.start()

    if args.img_input:
        env.reset()
        env = PixelObservationWrapper(env)

    agent = Agent(env, args.alpha, args.beta, args.hidden_dims, args.tau, args.batch_size,
                  args.gamma, args.d, args.warmup, args.max_size, args.c, args.sigma,
                  args.one_device, args.log_dir, args.checkpoint_dir, args.img_input,
                  args.in_channels, args.order, args.depth, args.multiplier,
                  args.action_embed_dim, args.hidden_dim, args.crop_dim)
    best_score = env.reward_range[0]
    load_weights(args.checkpoint_dir,
                 [agent.actor] , ['actor'])
    episodes = tqdm(range(args.n_episodes))

    for e in episodes:
        # resetting
        state = env.reset()
        if args.img_input:
            state_queue = deque(
                [preprocess_img(state['pixels'], args.crop_dim) for _ in range(args.order)],
                maxlen=args.order)
            state = torch.cat(list(state_queue), 1).cpu().numpy()
        done, score = False, 0

        while not done:
            action = agent.choose_action(state, rendering=True)
            state_, reward, _, _ = env.step(action)
            if isinstance(reward, np.ndarray):
                reward = reward[0]
            if args.img_input:
                state_queue.append(preprocess_img(state_['pixels'], args.crop_dim))
                state = torch.cat(list(state_queue), 1).cpu().numpy()

            # reset, log & render
            score += reward
            episodes.set_postfix({'Reward': reward})
            env.render()
        if score > best_score:
            best_score = score
        tqdm.write(f'Episode: {e + 1}/{args.n_episodes}, \
                    Episode Score: {score}, \
                    Best Score: {best_score}')
