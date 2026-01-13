import sys
import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ImpossibleGameEnv
from agents.policy.ppo_agent import PPOAgent
from training.config import PPO_CONFIG

def make_env():
    return ImpossibleGameEnv(max_steps=10000)

def train_ppo(config=PPO_CONFIG):
    models_dir = 'results/models'
    os.makedirs(models_dir, exist_ok=True)

    env = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=True, clip_obs=10.)

    agent = PPOAgent(env, **{k: config[k] for k in ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef']})
    
    print(f"Training PPO for {config['total_timesteps']} steps...")
    agent.update(total_timesteps=config['total_timesteps'])
    
    agent.save(os.path.join(models_dir, 'ppo_agent'))
    env.save(os.path.join(models_dir, 'vec_normalize.pkl'))
    print("Training complete")

if __name__ == "__main__":
    train_ppo()