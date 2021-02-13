import gym
import argparse
from tqdm import trange
from policy.agent import BlackJackAgent


parser = argparse.ArgumentParser(description='Black Jack Agents')
parser.add_argument('--method', type=str, default='lucky', help='The name of the policy you wish to evaluate')
parser.add_argument('--function', type=str, default='Q', help='The function to evaluate')
parser.add_argument('--n_episodes', type=int, default=500000, help='Number of episodes you wish to run for')
args = parser.parse_args()


def first_visit_monte_carlo():
    env = gym.make('Blackjack-v0')
    agent = BlackJackAgent(args.method, env, args.function)
    for _ in trange(args.n_episodes):
        state, done = env.reset(), False
        states, actions, rewards = [state], [], []
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            state = state_
        agent.update(rewards, states, actions)

    print(agent.value_function(21, 2, True))
    print(agent.q_function(16, 2, False, 0))


if __name__ == '__main__':
    first_visit_monte_carlo()
