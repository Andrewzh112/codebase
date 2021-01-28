"""
GDRL
"""
import numpy as np
from tqdm import trange


def softmax_decay(env, init_temp, min_temp, decay_ratio, n_episodes):
    pass


def upper_confidence_bound(env, c, n_episodes):
    Q = np.zeros(len(env.action_space.n))
    N = np.zeros(len(env.action_space.n))
    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)

    for e in trange(n_episodes):
        # avoiding zero division
        if e < len(Q):
            action = e
        # A_e = argmax[Q_e(a) + c * sqrt(ln(e) / N_e(a))]
        else:
            U = np.sqrt(c * np.log(e) / N)
            action = np.argmax(Q + U)

        # act and collect reward
        _, reward, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action
    return returns, Qe, actions
