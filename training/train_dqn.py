import sys
import os
import numpy as np
import gymnasium as gym
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. FIX: Importul corect al mediului
from environment.geometry_dash_env import ImpossibleGameEnv

# 2. Păstrăm importurile tale originale
from agents.deep.dqn_agent import DQNAgent
from training.config import DQN_CONFIG

# 3. Importăm wrapper-ele tale, dar folosim și gymnasium standard pentru siguranță
from environment.wrappers import FrameSkipWrapper, NormalizeObservation
# (RewardShapingWrapper este scos intenționat)

def train_dqn(config=DQN_CONFIG, render=False):
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)

    render_mode = "human" if render else None
    
    # Inițializare mediu
    raw_env = ImpossibleGameEnv(render_mode=render_mode, max_steps=config['max_steps'])
    
    # --- WRAPPERS ---
    # Folosim wrapper-ul tău de Skip
    env = FrameSkipWrapper(raw_env, skip=2)
    
    # FIX CRITIC: Folosim implementarea standard din Gym pentru Stack + Flatten
    # pentru a evita erorile de dimensiuni în rețeaua neurală.
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    env = gym.wrappers.FlattenObservation(env) 
    
    # Normalizare
    env = NormalizeObservation(env)
    
    print(f"Lidar Environment Initialized. Final Input Shape: {env.observation_space.shape}")
    
    # Inițializare Agent
    agent = DQNAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        learning_rate=config['learning_rate'],
        discount_factor=config['discount_factor'],
        epsilon=config['epsilon'],
        epsilon_decay=config['epsilon_decay'],
        epsilon_min=config['epsilon_min'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        target_update_freq=config['target_update_freq']
    )
    
    episode_rewards = []
    episode_scores = []
    best_avg_score = -float('inf')
    
    # Bara de progres
    pbar = tqdm(range(1, config['episodes'] + 1), desc="Training Lidar DQN")
    
    for episode in pbar:
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            
            if render: env.render()
        
        episode_rewards.append(episode_reward)
        current_score = info.get('score', 0)
        episode_scores.append(current_score)
        
        # --- LOGGING AVANSAT ---
        
        # 1. Actualizăm bara de progres la fiecare episod (Avg pe ultimele 50)
        avg_score_50 = np.mean(episode_scores[-50:]) if len(episode_scores) > 50 else np.mean(episode_scores)
        pbar.set_postfix({'Avg50': f'{avg_score_50:.1f}', 'Eps': f'{agent.epsilon:.3f}'})

        # 2. CHECKPOINT LA FIECARE 100 EPISOADE (Rămâne scris în consolă)
        if episode % 100 == 0:
            avg_score_100 = np.mean(episode_scores[-100:])
            max_score_100 = np.max(episode_scores[-100:])
            avg_reward_100 = np.mean(episode_rewards[-100:])
            
            tqdm.write(f"📊 CHECKPOINT Ep {episode}: "
                       f"Avg Score: {avg_score_100:.1f} | "
                       f"Max Score: {max_score_100} | "
                       f"Avg Reward: {avg_reward_100:.1f} | "
                       f"Epsilon: {agent.epsilon:.4f}")
            
            # Verificăm Best Model aici
            if avg_score_100 > best_avg_score and episode > 500:
                best_avg_score = avg_score_100
                tqdm.write(f"⭐ NEW BEST MODEL SAVED! (Avg Score: {best_avg_score:.1f})")
                agent.save('results/models/dqn_agent_best.pth')
        
        # 3. Salvare periodică de siguranță (la fiecare 1000)
        if episode % 1000 == 0:
            agent.save(f'results/models/dqn_agent_ep{episode}.pth')

    env.close()
    agent.save('results/models/dqn_agent.pth')
    
    np.save('results/logs/dqn_rewards.npy', episode_rewards)
    np.save('results/logs/dqn_scores.npy', episode_scores)

if __name__ == "__main__":
    train_dqn()