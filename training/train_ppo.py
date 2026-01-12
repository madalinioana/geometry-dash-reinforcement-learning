import sys
import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ImpossibleGameEnv
from agents.policy.ppo_agent import PPOAgent
from training.config import PPO_CONFIG

def make_env():
    """Helper pentru a crea mediul, necesar pentru DummyVecEnv."""
    return ImpossibleGameEnv(max_steps=10000)

def train_ppo(config=PPO_CONFIG):
    """Train PPO agent using Stable-Baselines3 with Normalization."""
    
    # 1. Creare directoare pentru salvare
    models_dir = 'results/models'
    os.makedirs(models_dir, exist_ok=True)

    # 2. Vectorizare și Normalizare (CRITIC PENTRU PPO)
    # PPO învață mult mai bine dacă inputurile sunt normalizate (medie 0, deviație 1)
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    agent = PPOAgent(
        env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef']
    )
    
    print(f"Training PPO agent for {config['total_timesteps']} steps...")
    print("Normalization enabled. Exploring with ent_coef:", config['ent_coef'])
    
    agent.update(total_timesteps=config['total_timesteps'])
    
    # 3. Salvare Model + Statistici Normalizare
    # Salvăm modelul
    agent.save(os.path.join(models_dir, 'ppo_agent'))
    
    # Salvăm statisticile mediului (media și deviația standard)
    # Fără asta, la testare agentul va vedea lumea 'greșit'
    env.save(os.path.join(models_dir, 'vec_normalize.pkl'))
    
    print("PPO training complete! Model and normalization stats saved.")

if __name__ == "__main__":
    train_ppo()