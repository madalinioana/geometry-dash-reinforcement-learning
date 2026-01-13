import gymnasium as gym
import numpy as np
from collections import deque

class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=2):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        return obs, total_reward, terminated, truncated, info

class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
        low = np.tile(env.observation_space.low, n_frames)
        high = np.tile(env.observation_space.high, n_frames)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob(), info
        
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, term, trunc, info
        
    def _get_ob(self):
        return np.concatenate(list(self.frames))

class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        return np.clip(obs, -1.0, 1.0)

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        reward += 0.05
        
        
        return obs, reward, terminated, truncated, info