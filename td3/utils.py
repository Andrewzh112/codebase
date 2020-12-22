import numpy as np
import torch
import numpy as np
from torchvision import transforms as T
from PIL import Image


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.counter = 0
        self.state_memory = np.empty((self.mem_size, *input_shape))
        self.new_state_memory = np.empty((self.mem_size, *input_shape))
        self.action_memory = np.empty((self.mem_size, n_actions))
        self.reward_memory = np.empty(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        state = self.state_memory[batch]
        state_ = self.new_state_memory[batch]
        done = self.terminal_memory[batch]
        reward = self.reward_memory[batch]
        action = self.action_memory[batch]

        return state, action, reward, state_, done


def preprocess_img(img, obs_size):
    """Taken from https://github.com/AaronAnima"""
    # to enlarge the occupancy of pendulum in background
    crop_ratio = 0.65
    # h, w, c -> c, h, w
    img = img.transpose((2, 0, 1))
    _, h, w = img.shape
    size = min(h, w)
    # manually center crop to square
    img_ts = img[:, (h//2) - (size//2): (h//2) + (size//2), (w//2) - (size//2): (w//2) + (size//2)]
    img_ts = np.ascontiguousarray(img_ts, dtype=np.float32) / 255.0
    img_ts = torch.from_numpy(img_ts)
    crop_size = int(crop_ratio * size)
    SetRange = T.Lambda(lambda X: 2 * X - 1.)
    transforms = T.Compose([T.ToPILImage(),
                        T.CenterCrop(crop_size),
                        T.Resize((obs_size, obs_size), interpolation=Image.CUBIC),
                        T.ToTensor(),
                        SetRange])
    img_ts = transforms(img_ts)
    return img_ts.unsqueeze(0)
