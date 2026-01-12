import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ImpossibleGameEnv
from agents.tabular.q_learning_agent import QLearningAgent
from agents.tabular.sarsa_agent import SARSAAgent
from agents.deep.dqn_agent import DQNAgent
from agents.policy.ppo_agent import PPOAgent
from evaluation.evaluate import evaluate_agent

sns.set_style("whitegrid")


def compare_agents(episodes=100):
    """Compare all trained agents and generate visualizations."""
    results = {}

    # Q-Learning
    if os.path.exists('results/models/q_learning_agent.pkl'):
        print("\nEvaluating Q-Learning...")
        env = ImpossibleGameEnv()
        agent = QLearningAgent(env.action_space, env.observation_space)
        agent.load('results/models/q_learning_agent.pkl')
        results['Q-Learning'] = evaluate_agent(agent, env, episodes=episodes)
        env.close()
    else:
        print("Q-Learning model not found.")

    # SARSA
    if os.path.exists('results/models/sarsa_agent.pkl'):
        print("\nEvaluating SARSA...")
        env = ImpossibleGameEnv()
        agent = SARSAAgent(env.action_space, env.observation_space)
        agent.load('results/models/sarsa_agent.pkl')
        results['SARSA'] = evaluate_agent(agent, env, episodes=episodes)
        env.close()
    else:
        print("SARSA model not found.")

    # DQN
    if os.path.exists('results/models/dqn_agent.pth'):
        print("\nEvaluating DQN...")
        env = ImpossibleGameEnv()
        agent = DQNAgent(env.action_space, env.observation_space)
        agent.load('results/models/dqn_agent.pth')
        results['DQN'] = evaluate_agent(agent, env, episodes=episodes)
        env.close()
    else:
        print("DQN model not found.")

    # PPO
    if os.path.exists('results/models/ppo_agent.zip'):
        print("\nEvaluating PPO...")
        env = ImpossibleGameEnv()
        agent = PPOAgent(env)
        agent.load('results/models/ppo_agent')
        results['PPO'] = evaluate_agent(agent, env, episodes=episodes)
        env.close()
    else:
        print("PPO model not found.")

    if not results:
        print("No trained models found! Train agents first.")
        return None

    # Create comparison table
    df = pd.DataFrame({
        'Agent': list(results.keys()),
        'Mean Score': [f"{r['mean_score']:.1f}" for r in results.values()],
        'Std Score': [f"{r['std_score']:.1f}" for r in results.values()],
        'Max Score': [r['max_score'] for r in results.values()],
        'Mean Reward': [f"{r['mean_reward']:.1f}" for r in results.values()],
        'Mean Length': [f"{r['mean_length']:.0f}" for r in results.values()]
    })

    print("\n" + "="*70)
    print("AGENT COMPARISON RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    os.makedirs('results/plots', exist_ok=True)
    df.to_csv('results/comparison_table.csv', index=False)

    # Generate comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    agents = list(results.keys())
    color_map = {a: colors[i % len(colors)] for i, a in enumerate(agents)}

    # Plot 1: Mean scores with error bars
    mean_scores = [results[a]['mean_score'] for a in agents]
    std_scores = [results[a]['std_score'] for a in agents]
    bars1 = axes[0, 0].bar(agents, mean_scores, yerr=std_scores, capsize=8,
                           color=[color_map[a] for a in agents], alpha=0.8)
    axes[0, 0].set_title('Mean Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for bar, score in zip(bars1, mean_scores):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_scores[agents.index(agents[bars1.index(bar)])],
                        f'{score:.0f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Mean rewards
    mean_rewards = [results[a]['mean_reward'] for a in agents]
    bars2 = axes[0, 1].bar(agents, mean_rewards, color=[color_map[a] for a in agents], alpha=0.8)
    axes[0, 1].set_title('Mean Reward Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Total Reward')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for bar, reward in zip(bars2, mean_rewards):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{reward:.0f}', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Score distribution (box plot)
    score_data = [results[a]['scores'] for a in agents]
    bp = axes[1, 0].boxplot(score_data, labels=agents, patch_artist=True)
    for patch, color in zip(bp['boxes'], [color_map[a] for a in agents]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_title('Score Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Plot 4: Mean episode length
    mean_lengths = [results[a]['mean_length'] for a in agents]
    bars4 = axes[1, 1].bar(agents, mean_lengths, color=[color_map[a] for a in agents], alpha=0.8)
    axes[1, 1].set_title('Mean Episode Length', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Steps')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for bar, length in zip(bars4, mean_lengths):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{length:.0f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Reinforcement Learning Agent Comparison\nGeometry Dash Environment',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/plots/agents_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to results/plots/agents_comparison.png")

    # Generate detailed statistics
    detailed_stats = []
    for agent_name, res in results.items():
        detailed_stats.append({
            'Agent': agent_name,
            'Mean Score': res['mean_score'],
            'Std Score': res['std_score'],
            'Min Score': np.min(res['scores']),
            'Max Score': res['max_score'],
            'Median Score': np.median(res['scores']),
            'Mean Reward': res['mean_reward'],
            'Std Reward': res['std_reward'],
            'Mean Length': res['mean_length']
        })

    detailed_df = pd.DataFrame(detailed_stats)
    detailed_df.to_csv('results/detailed_comparison.csv', index=False)
    print("Detailed stats saved to results/detailed_comparison.csv")

    return results


if __name__ == "__main__":
    compare_agents(episodes=100)
