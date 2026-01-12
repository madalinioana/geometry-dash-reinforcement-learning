import sys
import os
import numpy as np
import gymnasium as gym
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.geometry_dash_env import ImpossibleGameEnv
from agents.deep.dqn_agent import DQNAgent
from environment.wrappers import FrameSkipWrapper, NormalizeObservation

# --- CONFIGURARE PENTRU FINE-TUNING ---
FINE_TUNE_CONFIG = {
    'episodes': 5000,           # Doar 5000 de episoade de rafinare
    'max_steps': 5000,          # Îl lăsăm să meargă mai departe dacă poate
    'learning_rate': 0.00001,   # FOARTE MIC! (10x mai mic ca înainte) - Doar ajustări fine
    'discount_factor': 0.995,
    'epsilon': 0.15,            # Începem cu puțină explorare
    'epsilon_decay': 0.999,     # Scade rapid spre 0
    'epsilon_min': 0.001,       # Aproape zero explorare la final
    'buffer_size': 200000,
    'batch_size': 128,
    'target_update_freq': 4000
}

def continue_training():
    os.makedirs('results/models', exist_ok=True)
    
    # 1. Mediu
    raw_env = ImpossibleGameEnv(render_mode=None, max_steps=FINE_TUNE_CONFIG['max_steps'])
    env = FrameSkipWrapper(raw_env, skip=2)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    env = gym.wrappers.FlattenObservation(env) 
    env = NormalizeObservation(env)
    
    # 2. Agent
    agent = DQNAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        learning_rate=FINE_TUNE_CONFIG['learning_rate'],
        epsilon=FINE_TUNE_CONFIG['epsilon'],
        epsilon_decay=FINE_TUNE_CONFIG['epsilon_decay'],
        epsilon_min=FINE_TUNE_CONFIG['epsilon_min'],
        buffer_size=FINE_TUNE_CONFIG['buffer_size'],
        batch_size=FINE_TUNE_CONFIG['batch_size']
    )
    
    # 3. ÎNCĂRCĂM CREIERUL ANTRENAT
    path = 'results/models/dqn_agent_best.pth'
    if os.path.exists(path):
        print(f"LOADING TRAINED MODEL: {path}")
        agent.load(path)
        # Resetăm epsilon manual pentru fine-tuning (load-ul poate suprascrie)
        agent.epsilon = FINE_TUNE_CONFIG['epsilon']
    else:
        print("EROARE: Nu am găsit modelul antrenat! Rulează train_dqn.py întâi.")
        return

    episode_scores = []
    best_avg_score = -float('inf')
    
    pbar = tqdm(range(1, FINE_TUNE_CONFIG['episodes'] + 1), desc="Fine-Tuning DQN")
    
    for episode in pbar:
        obs, info = env.reset()
        episode_score = 0
        done = False
        
        while not done:
            # Aici epsilon va scădea rapid, lăsând agentul să joace "pe bune"
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            
        current_score = info.get('score', 0)
        episode_scores.append(current_score)
        
        # Logging
        if episode % 50 == 0:
            avg_score = np.mean(episode_scores[-50:])
            max_score = np.max(episode_scores[-50:])
            pbar.set_postfix({'Avg': f'{avg_score:.1f}', 'Max': f'{max_score}', 'Eps': f'{agent.epsilon:.4f}'})
            
            # Salvăm un model "Refined" separat
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save('results/models/dqn_agent_refined.pth')
                tqdm.write(f"🚀 NEW RECORD: Avg {avg_score:.1f} | Max {max_score}")

    env.close()
    print("Fine-tuning complete.")

if __name__ == "__main__":
    continue_training()