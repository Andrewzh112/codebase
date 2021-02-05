"""
https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/DQN/utils.py
"""

from collections import deque
from PIL import Image, ImageOps
import gym
import numpy as np


class RepeatAction(gym.Wrapper):
    def __init__(self, env, n_repeats, clip_rewards, no_ops, fire_first):
        super().__init__(env)
        self.n_repeats = n_repeats
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros((2, *self.shape))
        self.clip_rewards = clip_rewards
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        done, score = False, 0
        for i in range(self.n_repeats):
            observation, reward, done, info = self.env.step(action)
            if self.clip_rewards:
                reward =  max(-1, min(reward, 1))
            score += reward
            self.frame_buffer[i % 2] = observation
            if done:
                break
        max_frame = np.maximum(*self.frame_buffer)
        return max_frame, score, done, info

    def reset(self):
        observation = self.env.reset()
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            observation, _, _, _ = self.env.step(1)
        self.frame_buffer = np.zeros((2, *self.shape))
        self.frame_buffer[0] = observation
        return observation


class Preprocess(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        processed_observation = observation.astype(np.uint8)
        processed_observation = Image.fromarray(processed_observation)
        processed_observation = ImageOps.grayscale(processed_observation)
        processed_observation = processed_observation.resize(self.shape[1:])
        processed_observation = np.expand_dims(np.asarray(processed_observation), axis=0)
        return processed_observation / 255.0


class FrameStacker(gym.ObservationWrapper):
    def __init__(self, env, n_repeats):
        super().__init__(env)
        self.n_repeats = n_repeats
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(n_repeats, axis=0),
            env.observation_space.high.repeat(n_repeats, axis=0),
            dtype=np.float32)
        self.stack = deque(maxlen=n_repeats)
        self.observation_shape = self.observation_space.low.shape

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.n_repeats):
            self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_shape)

    def observation(self, observation):
        self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_shape)

def processed_atari(env_name, shape=84, input_channels=1, n_repeats=4, clip_rewards=False, no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatAction(env, n_repeats, clip_rewards, no_ops, fire_first)
    env = Preprocess(env, (shape, shape, input_channels))
    env = FrameStacker(env, n_repeats)
    return env
