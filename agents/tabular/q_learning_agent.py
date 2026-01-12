import numpy as np
import pickle
from agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent cu state discretization.
    """
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
        
        # Discretizare observatii (folosim doar primele 3 features)
        self.state_bins = [bins, bins, 2]  # y, vel_y, on_ground
        self.q_table = {}
    
    def discretize_state(self, observation):
        """Discretizeaza observatia continua intr-un state."""
        # Folosim doar player_y, vel_y, on_ground
        player_y = observation[0]
        vel_y = observation[1]
        on_ground = int(observation[2])
        
        # Discretizare
        y_bin = int(np.clip(player_y * self.bins, 0, self.bins - 1))
        vel_bin = int(np.clip((vel_y + 1) * self.bins / 2, 0, self.bins - 1))
        
        return (y_bin, vel_bin, on_ground)
    
    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        return self.q_table[state][action]
    
    def select_action(self, observation, training=True):
        """Epsilon-greedy action selection."""
        state = self.discretize_state(observation)
        
        if training and np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            if state not in self.q_table:
                return self.action_space.sample()
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done, training=True):
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)
        
        current_q = self.get_q_value(state, action)
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max([self.get_q_value(next_state, a) 
                                for a in range(self.action_space.n)])
            target_q = reward + self.gamma * max_next_q
        
        # Q-Learning update
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        
        self.q_table[state][action] += self.lr * (target_q - current_q)
        
        # Decay epsilon
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, f)
        print(f"Q-Learning agent saved to {path}")
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
        print(f"Q-Learning agent loaded from {path}")