import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return tuple(np.array(x, dtype=np.float32 if i != 1 else np.int64) 
                    for i, x in enumerate([states, actions, rewards, next_states, dones]))
    
    def __len__(self):
        return len(self.buffer)