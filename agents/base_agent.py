from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Clasa de bazw pentru toti agentii.
    """
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
    
    @abstractmethod
    def select_action(self, observation, training=True):
        """actiune bazata pe observatie."""
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs):
        """Actualizeazw parametrii agentului."""
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass