import numpy as np
import pickle
from agents.base_agent import BaseAgent


class SARSAAgent(BaseAgent):
    def __init__(self, action_space, observation_space, 
                 learning_rate=0.1, discount_factor=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 bins=10):
        super().__init__(action_space, observation_space)
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.bins = bins
        
        self.q_table = {}
    
    def discretize_state(self, observation):
        player_y = observation[0]
        vel_y = observation[1]
        on_ground = int(observation[2])
        lidar_types = observation[3:33]
        lidar_heights = observation[33:63]
        
        y_bin = int(np.clip(player_y * self.bins, 0, self.bins - 1))
        vel_bin = int(np.clip((vel_y + 1) * self.bins / 2, 0, self.bins - 1))
        
        near_danger_type = 0
        near_height = 0
        mid_danger = 0
        
        for i in range(min(5, len(lidar_types))):
            if lidar_types[i] == 2:
                near_danger_type = 2
                near_height = int(np.clip(lidar_heights[i] * 4, 0, 3))
                break
            elif lidar_types[i] == 1 and near_danger_type == 0:
                near_danger_type = 1
                near_height = int(np.clip(lidar_heights[i] * 4, 0, 3))
        
        for i in range(5, min(10, len(lidar_types))):
            if lidar_types[i] == 2:
                mid_danger = 1
                break
        
        return (y_bin, vel_bin, on_ground, near_danger_type, near_height, mid_danger)
    
    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        return self.q_table[state][action]
    
    def select_action(self, observation, training=True):
        state = self.discretize_state(observation)
        if (training and np.random.random() < self.epsilon) or state not in self.q_table:
            return self.action_space.sample()
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, next_action, done, training=True):
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)

        current_q = self.q_table[state][action]
        target_q = reward if done else reward + self.gamma * self.get_q_value(next_state, next_action)
        self.q_table[state][action] += self.lr * (target_q - current_q)

        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'epsilon': self.epsilon}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
