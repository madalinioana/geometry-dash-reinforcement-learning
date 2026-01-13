import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.geometry_dash_env import ImpossibleGameEnv
from agents.tabular.q_learning_agent import QLearningAgent
from training.config import Q_LEARNING_CONFIG


def train_q_learning(config=Q_LEARNING_CONFIG, render=False):
    
    render_mode = "human" if render else None
    env = ImpossibleGameEnv(render_mode=render_mode, max_steps=config['max_steps'])
    
    agent = QLearningAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        learning_rate=config['learning_rate'],
        discount_factor=config['discount_factor'],
        epsilon=config['epsilon'],
        epsilon_decay=config['epsilon_decay'],
        epsilon_min=config['epsilon_min'],
        bins=config['bins']
    )
    
    episode_rewards = []
    episode_scores = []
    
    for episode in tqdm(range(config['episodes']), desc="Training Q-Learning"):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            lidar_types = obs[3:33]
            on_ground = int(obs[2]) == 1
            has_nearby_danger = any(lidar_types[i] == 2 for i in range(5))
            has_mid_danger = any(lidar_types[i] == 2 for i in range(5, 10))
            
            if action == 1 and on_ground:
                if not has_nearby_danger and not has_mid_danger:
                    reward -= 1.0
                elif not has_nearby_danger and has_mid_danger:
                    reward -= 0.3
            elif action == 0 and on_ground and not has_nearby_danger:
                reward += 0.1
            
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_scores.append(info['score'])
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_score = np.mean(episode_scores[-100:])
            print(f"Episode {episode+1}: Reward={avg_reward:.1f}, Score={avg_score:.0f}, Eps={agent.epsilon:.3f}")
    
    env.close()
    
    os.makedirs('results/models', exist_ok=True)
    agent.save('results/models/q_learning_agent.pkl')
    
    np.save('results/logs/q_learning_rewards.npy', episode_rewards)
    np.save('results/logs/q_learning_scores.npy', episode_scores)
    
    print("Training complete")
    
    return episode_rewards, episode_scores


if __name__ == "__main__":
    rewards, scores = train_q_learning(render=False)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Q-Learning: Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(scores)
    plt.title('Q-Learning: Episode Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('results/plots/q_learning_training.png')
    plt.show()