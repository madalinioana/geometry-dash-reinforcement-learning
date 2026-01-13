import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ImpossibleGameEnv
from agents.tabular.sarsa_agent import SARSAAgent
from training.config import SARSA_CONFIG


def train_sarsa(config=SARSA_CONFIG, render=False):

    render_mode = "human" if render else None
    env = ImpossibleGameEnv(render_mode=render_mode, max_steps=config['max_steps'])

    agent = SARSAAgent(
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

    for episode in tqdm(range(config['episodes']), desc="Training SARSA"):
        obs, info = env.reset()
        action = agent.select_action(obs, training=True)
        episode_reward = 0
        done = False

        while not done:
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

            next_action = agent.select_action(next_obs, training=True)

            agent.update(obs, action, reward, next_obs, next_action, done)

            obs = next_obs
            action = next_action
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
    os.makedirs('results/logs', exist_ok=True)
    agent.save('results/models/sarsa_agent.pkl')

    np.save('results/logs/sarsa_rewards.npy', episode_rewards)
    np.save('results/logs/sarsa_scores.npy', episode_scores)

    print("Training complete")

    return episode_rewards, episode_scores


if __name__ == "__main__":
    rewards, scores = train_sarsa(render=False)

    os.makedirs('results/plots', exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3)
    window = 50
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, linewidth=2)
    plt.title('SARSA: Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(scores, alpha=0.3)
    if len(scores) >= window:
        smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, linewidth=2)
    plt.title('SARSA: Episode Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/plots/sarsa_training.png', dpi=300)
    plt.show()
