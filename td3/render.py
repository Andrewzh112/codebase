import gym
from tqdm import tqdm

from td3.agent import Agent
from td3.main import args
from networks.utils import load_weights


if __name__ == '__main__':
    args.checkpoint_dir += f'/{args.env_name}_td3.pth'
    # env & agent
    env = gym.make(args.env_name)
    agent = Agent(env, args.alpha, args.beta, args.hidden_dims, args.tau, args.batch_size,
                  args.gamma, args.d, 0, args.max_size, args.c, args.sigma,
                  args.one_device, args.log_dir, args.checkpoint_dir)
    best_score = env.reward_range[0]
    load_weights(args.checkpoint_dir,
                 [agent.actor] , ['actor'])
    episodes = tqdm(range(args.n_episodes))
    for e in episodes:
        # resetting
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)

            # reset, log & render
            score += reward
            state = state_
            episodes.set_postfix({'Reward': reward})
            env.render()
        if score > best_score:
            best_score = score
        tqdm.write(f'Episode: {e + 1}/{args.n_episodes}, \
                    Episode Score: {score}, \
                    Best Score: {best_score}')
