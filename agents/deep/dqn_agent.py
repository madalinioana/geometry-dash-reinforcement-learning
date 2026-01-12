import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent
from agents.deep.replay_buffer import ReplayBuffer

class DQN(nn.Module):
    """Deep Q-Network cu capacitate crescută pentru pattern-uri complexe."""
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent(BaseAgent):
    """Double DQN Agent optimizat pentru Geometry Dash."""
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
        print(f"DQN Agent initialized on device: {self.device}")
        
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Input dim este determinat automat din mediul Flattened
        input_dim = observation_space.shape[0]
        output_dim = action_space.n
        
        self.q_network = DQN(input_dim, output_dim).to(self.device)
        self.target_network = DQN(input_dim, output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() # Target net nu se antrenează niciodată direct
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        # Huber Loss (SmoothL1) e mai stabil pentru DQN decat MSE
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
        
        # Fixare dimensiuni pentru broadcasting corect [Batch, 1]
        rewards = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        dones = torch.FloatTensor(dones).view(-1, 1).to(self.device)
        
        # 1. Calculăm Q(s, a) curent
        # gather ia valoarea Q corespunzătoare acțiunii luate
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 2. Calculăm Target Q folosind Double DQN
        with torch.no_grad():
            # a) Rețeaua curentă alege cea mai bună acțiune viitoare (Argmax)
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            
            # b) Rețeaua target evaluează acea acțiune
            next_q = self.target_network(next_states).gather(1, next_actions)
            
            # c) Formula Bellman
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 3. Calculăm Loss și facem Backprop
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping pentru stabilitate (evită explozia gradientilor)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 4. Actualizări periodice
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
        # map_location asigură că încarcă pe CPU dacă a fost salvat pe GPU și vice-versa
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint['epsilon']
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])