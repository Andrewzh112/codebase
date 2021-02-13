import gym
import argparse
from tqdm import trange
from policy.agent import CartPoleNoob


parser = argparse.ArgumentParser(description='Cartpole Agents')
parser.add_argument('--method', type=str, default='egreedy', help='The name of the policy you wish to evaluate')
parser.add_argument('--function', type=str, default='Q', help='The function to evaluate')
parser.add_argument('--n_episodes', type=int, default=500000, help='Number of episodes you wish to run for')
args = parser.parse_args()


def td():
    env = gym.make('CartPole-v0')
    agent = CartPoleNoob(args.method, env, args.function)
    for _ in trange(args.n_episodes):
        state, done = env.reset(), False
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            agent.update(state, action, reward, state_)
            state = state_
    print(agent.values)

if __name__ == '__main__':
    td()
