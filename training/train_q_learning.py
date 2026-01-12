import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ImpossibleGameEnv
from agents.tabular.q_learning_agent import QLearningAgent
from training.config import Q_LEARNING_CONFIG


def train_q_learning(config=Q_LEARNING_CONFIG, render=False):
    
    # Create environment
    render_mode = "human" if render else None
    env = ImpossibleGameEnv(render_mode=render_mode, max_steps=config['max_steps'])
    
    # Create agent
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
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    
    # Training loop
    for episode in tqdm(range(config['episodes']), desc="Training Q-Learning"):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(info['score'])
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_score = np.mean(episode_scores[-100:])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}, Avg Score = {avg_score:.0f}, Epsilon = {agent.epsilon:.3f}")
    
    env.close()
    
    os.makedirs('results/models', exist_ok=True)
    agent.save('results/models/q_learning_agent.pkl')
    
    # Save metrics
    np.save('results/logs/q_learning_rewards.npy', episode_rewards)
    np.save('results/logs/q_learning_scores.npy', episode_scores)
    
    print(f"Training complete! Saved to results/models/q_learning_agent.pkl")
    
    return episode_rewards, episode_scores


if __name__ == "__main__":
    rewards, scores = train_q_learning(render=False)
    
    # Plot results
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