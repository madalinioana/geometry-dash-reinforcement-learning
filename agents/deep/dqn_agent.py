import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent
from agents.deep.replay_buffer import ReplayBuffer

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent(BaseAgent):
    def __init__(self, action_space, observation_space,
                 learning_rate=3e-4,
                 discount_factor=0.99,
                 epsilon=1.0, 
                 epsilon_decay=0.9998,
                 epsilon_min=0.05,
                 buffer_size=150000, 
                 batch_size=128,
                 target_update_freq=3000):
        
        super().__init__(action_space, observation_space)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        input_dim = observation_space.shape[0]
        output_dim = action_space.n
        
        self.q_network = DQN(input_dim, output_dim).to(self.device)
        self.target_network = DQN(input_dim, output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.update_count = 0
    
    def select_action(self, observation, training=True):
        if training and np.random.random() < self.epsilon:
            return self.action_space.sample()
        
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        rewards = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        dones = torch.FloatTensor(dones).view(-1, 1).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            
            next_q = self.target_network(next_states).gather(1, next_actions)
            
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint['epsilon']
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])