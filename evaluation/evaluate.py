import gymnasium as gym
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ImpossibleGameEnv
from agents.tabular.q_learning_agent import QLearningAgent
from agents.tabular.sarsa_agent import SARSAAgent
from agents.deep.dqn_agent import DQNAgent
from agents.policy.ppo_agent import PPOAgent


def evaluate_agent(agent, env, episodes=100, render=False, verbose=True):
    """Evaluate an agent over N episodes."""

    episode_rewards = []
    episode_scores = []
    episode_lengths = []

    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_scores.append(info['score'])
        episode_lengths.append(episode_length)

        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{episodes}: Score = {info['score']}, Reward = {episode_reward:.2f}")

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_score': np.mean(episode_scores),
        'std_score': np.std(episode_scores),
        'mean_length': np.mean(episode_lengths),
        'max_score': np.max(episode_scores),
        'min_score': np.min(episode_scores),
        'median_score': np.median(episode_scores),
        'rewards': episode_rewards,
        'scores': episode_scores,
        'lengths': episode_lengths
    }


def main():
    """Evaluate all trained agents."""

    results = {}

    # Evaluate Q-Learning
    print("\n" + "="*50)
    print("Evaluating Q-Learning")
    print("="*50)
    env = ImpossibleGameEnv()
    agent_q = QLearningAgent(env.action_space, env.observation_space)

    if os.path.exists('results/models/q_learning_agent.pkl'):
        agent_q.load('results/models/q_learning_agent.pkl')
        results['Q-Learning'] = evaluate_agent(agent_q, env, episodes=100)
        print(f"\nQ-Learning Results:")
        print(f"  Mean Score: {results['Q-Learning']['mean_score']:.2f} +/- {results['Q-Learning']['std_score']:.2f}")
        print(f"  Max Score: {results['Q-Learning']['max_score']}")
        print(f"  Mean Reward: {results['Q-Learning']['mean_reward']:.2f}")
    else:
        print("Q-Learning model not found!")
    env.close()

    # Evaluate SARSA
    print("\n" + "="*50)
    print("Evaluating SARSA")
    print("="*50)
    env = ImpossibleGameEnv()
    agent_sarsa = SARSAAgent(env.action_space, env.observation_space)

    if os.path.exists('results/models/sarsa_agent.pkl'):
        agent_sarsa.load('results/models/sarsa_agent.pkl')
        results['SARSA'] = evaluate_agent(agent_sarsa, env, episodes=100)
        print(f"\nSARSA Results:")
        print(f"  Mean Score: {results['SARSA']['mean_score']:.2f} +/- {results['SARSA']['std_score']:.2f}")
        print(f"  Max Score: {results['SARSA']['max_score']}")
        print(f"  Mean Reward: {results['SARSA']['mean_reward']:.2f}")
    else:
        print("SARSA model not found!")
    env.close()

    # Evaluate DQN
    print("\n" + "="*50)
    print("Evaluating DQN")
    print("="*50)
    env = ImpossibleGameEnv()
    agent_dqn = DQNAgent(env.action_space, env.observation_space)

    if os.path.exists('results/models/dqn_agent.pth'):
        agent_dqn.load('results/models/dqn_agent.pth')
        results['DQN'] = evaluate_agent(agent_dqn, env, episodes=100)
        print(f"\nDQN Results:")
        print(f"  Mean Score: {results['DQN']['mean_score']:.2f} +/- {results['DQN']['std_score']:.2f}")
        print(f"  Max Score: {results['DQN']['max_score']}")
        print(f"  Mean Reward: {results['DQN']['mean_reward']:.2f}")
    else:
        print("DQN model not found!")
    env.close()

    # Evaluate PPO
    print("\n" + "="*50)
    print("Evaluating PPO")
    print("="*50)
    env = ImpossibleGameEnv()

    if os.path.exists('results/models/ppo_agent.zip'):
        agent_ppo = PPOAgent(env)
        agent_ppo.load('results/models/ppo_agent')
        results['PPO'] = evaluate_agent(agent_ppo, env, episodes=100)
        print(f"\nPPO Results:")
        print(f"  Mean Score: {results['PPO']['mean_score']:.2f} +/- {results['PPO']['std_score']:.2f}")
        print(f"  Max Score: {results['PPO']['max_score']}")
        print(f"  Mean Reward: {results['PPO']['mean_reward']:.2f}")
    else:
        print("PPO model not found!")
    env.close()

    # Print summary
    if results:
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"{'Agent':<15} {'Mean Score':<15} {'Max Score':<12} {'Mean Reward':<15}")
        print("-"*70)
        for agent_name, res in results.items():
            print(f"{agent_name:<15} {res['mean_score']:.1f} +/- {res['std_score']:.1f}   "
                  f"{res['max_score']:<12} {res['mean_reward']:.1f}")
        print("="*70)

    return results


if __name__ == "__main__":
    main()
