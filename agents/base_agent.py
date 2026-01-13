from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
    
    @abstractmethod
    def select_action(self, observation, training=True):
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass