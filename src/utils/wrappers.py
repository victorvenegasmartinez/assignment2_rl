import numpy as np
import gymnasium as gym
import ale_py
from utils.viewer import SimpleImageViewer
from collections import deque


class MaxAndSkipEnv(gym.Wrapper):
    """
    Takes a max pool over the last n states

    Attribution:
        This class was originally created for CS 294 at
        UC Berkeley - [https://github.com/berkeleydeeprlcourse/homework/blob/dde95f4e126e14a343a53efe25d1c2205854ea3a/hw3/dqn_utils.py#L174]
    """

    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs, info = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class PreproWrapper(gym.Wrapper):
    """
    Wrapper for Pong to apply preprocessing
    Stores the state into variable self.obs
    """

    def __init__(self, env, prepro, shape, overwrite_render=True, high=255):
        """
        Args:
            env: (gym env)
            prepro: (function) to apply to a state for preprocessing
            shape: (list) shape of obs after prepro
            overwrite_render: (bool) if True, render is overwriten to vizualise effect of prepro
            grey_scale: (bool) if True, assume grey scale, else black and white
            high: (int) max value of state after prepro
        """
        super(PreproWrapper, self).__init__(env)
        self.overwrite_render = overwrite_render
        self.viewer = None
        self.prepro = prepro
        self.observation_space = gym.spaces.Box(
            low=0, high=high, shape=shape, dtype=np.uint8
        )
        self.high = high

    def step(self, action):
        """
        Overwrites _step function from environment to apply preprocess
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = self.prepro(obs)
        return self.obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.obs = self.prepro(self.env.reset())
        return self.obs

    def _render(self, mode="human", close=False):
        """
        Overwrite _render function to vizualize preprocessing
        """

        if self.overwrite_render:
            if close:
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                return
            img = self.obs
            if mode == "rgb_array":
                return img
            elif mode == "human":
                from gymnasium.envs.classic_control import rendering

                if self.viewer is None:
                    self.viewer = SimpleImageViewer()
                self.viewer.imshow(img)

        else:
            super(PreproWrapper, self)._render(mode, close)
