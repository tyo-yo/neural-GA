import ctypes
import multiprocessing
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import time
import json
import logging
from datetime import datetime
logger = logging.getLogger(__name__)


class SharedNoiseTable(object):
    def __init__(self, seed=123, size=250000000):
        self.seed = seed
        self.size = size
        self._shared_mem = multiprocessing.Array(ctypes.c_float, self.size)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        # 64-bit to 32-bit conversion here
        self.noise[:] = np.random.RandomState(self.seed).randn(self.size)
        logger.info('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)


class ConvertToState():
    """Convert observation into state for model input."""

    def __init__(self):
        self.observation_tensor_list = []
        self.tsfm = T.Compose([
            T.ToPILImage(),
            T.Resize((84, 84)),
            T.Grayscale(),
            T.ToTensor(),
            T.Normalize(mean=(0.,), std=(0.5,))
        ])

    def convert(self, observation):
        """Return model input state tensor, shape = (1, 4, 84, 84)."""
        obsevation_tensor = self.tsfm(observation)
        if len(self.observation_tensor_list) < 4:
            self.observation_tensor_list = [obsevation_tensor] * 4
        else:
            self.observation_tensor_list.append(obsevation_tensor)
            self.observation_tensor_list.pop(0)
        return torch.cat(self.observation_tensor_list).unsqueeze(dim=0)

    def reset(self):
        self.observation_tensor_list = []


def rollout(model, env, max_steps=1000, rendering=False, return_steps=False,
            episode=1):
    """Excecute episode and return average reward."""
    observation = env.reset()
    convert_to_state = ConvertToState()
    total_reward = 0.0
    total_steps = 0
    for e in range(episode):
        for i in range(max_steps):
            state = convert_to_state.convert(observation)
            action = model.choose_action(state)
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            if rendering:
                render(env)
            if done:
                env.reset()
                convert_to_state.reset()
                total_steps += i
                break

    if return_steps:
        return total_reward / episode, float(i) / episode
    else:
        return total_reward / episode


def render(env, name='Render', rescale=1, sleep_time=0.01):
    frame = env.render(mode='rgb_array')
    frame = frame[:, :, ::-1]
    h = int(frame.shape[1] * rescale)
    w = int(frame.shape[0] * rescale)
    if rescale != 1:
        frame = cv2.resize(frame, (h, w))
    cv2.imshow(name, frame)
    cv2.waitKey(1)
    time.sleep(sleep_time)

def save_json(code, prefix=''):
    time_s = datetime.now().strftime('%m%d-%H%M')
    with open(prefix + 'model_{}_{:d}.json'.format(time_s, int(code['fitness'])), 'w') as f:
        json.dump(code, f)
