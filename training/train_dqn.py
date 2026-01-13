import sys
import os
import numpy as np
import gymnasium as gym
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.geometry_dash_env import ImpossibleGameEnv

from agents.deep.dqn_agent import DQNAgent
from training.config import DQN_CONFIG

from environment.wrappers import FrameSkipWrapper, NormalizeObservation

def train_dqn(config=DQN_CONFIG, render=False):
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)

    render_mode = "human" if render else None
    
    raw_env = ImpossibleGameEnv(render_mode=render_mode, max_steps=config['max_steps'])
    
    env = FrameSkipWrapper(raw_env, skip=2)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    env = gym.wrappers.FlattenObservation(env)
    env = NormalizeObservation(env)
    
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
        
        avg_score_50 = np.mean(episode_scores[-50:]) if len(episode_scores) > 50 else np.mean(episode_scores)
        pbar.set_postfix({'Avg50': f'{avg_score_50:.1f}', 'Eps': f'{agent.epsilon:.3f}'})

        if episode % 100 == 0:
            avg_score_100 = np.mean(episode_scores[-100:])
            max_score_100 = np.max(episode_scores[-100:])
            avg_reward_100 = np.mean(episode_rewards[-100:])
            
            tqdm.write(f"Ep {episode}: Avg={avg_score_100:.1f} Max={max_score_100} Reward={avg_reward_100:.1f} Eps={agent.epsilon:.4f}")
            
            if avg_score_100 > best_avg_score and episode > 500:
                best_avg_score = avg_score_100
                agent.save('results/models/dqn_agent_best.pth')
        
        if episode % 1000 == 0:
            agent.save(f'results/models/dqn_agent_ep{episode}.pth')

    env.close()
    agent.save('results/models/dqn_agent.pth')
    
    np.save('results/logs/dqn_rewards.npy', episode_rewards)
    np.save('results/logs/dqn_scores.npy', episode_scores)

if __name__ == "__main__":
    train_dqn()