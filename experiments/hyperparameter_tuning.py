"""
Hyperparameter tuning experiments for RL agents.
Runs multiple configurations and compares results.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from tqdm import tqdm
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ImpossibleGameEnv
from agents.tabular.q_learning_agent import QLearningAgent
from agents.tabular.sarsa_agent import SARSAAgent
from agents.deep.dqn_agent import DQNAgent


def run_experiment(agent_class, agent_name, config, episodes=1000, eval_episodes=50):
    """Run a single experiment with given configuration."""
    env = ImpossibleGameEnv(max_steps=config.get('max_steps', 10000))

    agent = agent_class(
        action_space=env.action_space,
        observation_space=env.observation_space,
        **{k: v for k, v in config.items() if k not in ['episodes', 'max_steps']}
    )

    episode_scores = []

    for episode in range(episodes):
        obs, info = env.reset()

        if agent_name == 'SARSA':
            action = agent.select_action(obs, training=True)

        done = False
        while not done:
            if agent_name == 'SARSA':
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_action = agent.select_action(next_obs, training=True)
                agent.update(obs, action, reward, next_obs, next_action, done)
                obs = next_obs
                action = next_action
            else:
                action = agent.select_action(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs

        episode_scores.append(info['score'])

    env.close()

    # Evaluation phase
    env = ImpossibleGameEnv()
    eval_scores = []

    for _ in range(eval_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        eval_scores.append(info['score'])

    env.close()

    return {
        'training_scores': episode_scores,
        'eval_mean': np.mean(eval_scores),
        'eval_std': np.std(eval_scores),
        'eval_max': np.max(eval_scores),
        'final_100_mean': np.mean(episode_scores[-100:]) if len(episode_scores) >= 100 else np.mean(episode_scores)
    }


def run_learning_rate_experiment():
    """Experiment: Effect of learning rate on Q-Learning."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Learning Rate Impact (Q-Learning)")
    print("="*60)

    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
    results = []

    for lr in tqdm(learning_rates, desc="Testing learning rates"):
        config = {
            'learning_rate': lr,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.9995,
            'epsilon_min': 0.01,
            'bins': 10
        }

        result = run_experiment(QLearningAgent, 'Q-Learning', config, episodes=1000)
        result['learning_rate'] = lr
        results.append(result)
        print(f"  LR={lr}: Eval Score = {result['eval_mean']:.1f} +/- {result['eval_std']:.1f}")

    return results


def run_discount_factor_experiment():
    """Experiment: Effect of discount factor (gamma)."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Discount Factor Impact (Q-Learning)")
    print("="*60)

    gammas = [0.8, 0.9, 0.95, 0.99, 0.999]
    results = []

    for gamma in tqdm(gammas, desc="Testing discount factors"):
        config = {
            'learning_rate': 0.1,
            'discount_factor': gamma,
            'epsilon': 1.0,
            'epsilon_decay': 0.9995,
            'epsilon_min': 0.01,
            'bins': 10
        }

        result = run_experiment(QLearningAgent, 'Q-Learning', config, episodes=1000)
        result['discount_factor'] = gamma
        results.append(result)
        print(f"  Gamma={gamma}: Eval Score = {result['eval_mean']:.1f} +/- {result['eval_std']:.1f}")

    return results


def run_exploration_experiment():
    """Experiment: Effect of exploration rate decay."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Exploration Decay Impact (Q-Learning)")
    print("="*60)

    epsilon_decays = [0.99, 0.995, 0.999, 0.9995, 0.9999]
    results = []

    for decay in tqdm(epsilon_decays, desc="Testing epsilon decays"):
        config = {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': decay,
            'epsilon_min': 0.01,
            'bins': 10
        }

        result = run_experiment(QLearningAgent, 'Q-Learning', config, episodes=1000)
        result['epsilon_decay'] = decay
        results.append(result)
        print(f"  Decay={decay}: Eval Score = {result['eval_mean']:.1f} +/- {result['eval_std']:.1f}")

    return results


def run_tabular_comparison():
    """Experiment: Compare Q-Learning vs SARSA with same hyperparameters."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Q-Learning vs SARSA Comparison")
    print("="*60)

    config = {
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.9995,
        'epsilon_min': 0.01,
        'bins': 10
    }

    results = {}

    print("Training Q-Learning...")
    results['Q-Learning'] = run_experiment(QLearningAgent, 'Q-Learning', config, episodes=2000)
    print(f"  Q-Learning: Eval Score = {results['Q-Learning']['eval_mean']:.1f}")

    print("Training SARSA...")
    results['SARSA'] = run_experiment(SARSAAgent, 'SARSA', config, episodes=2000)
    print(f"  SARSA: Eval Score = {results['SARSA']['eval_mean']:.1f}")

    return results


def plot_experiment_results(exp_name, results, param_name, save_dir='results/experiments'):
    """Generate visualization for experiment results."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Final evaluation scores
    params = [r[param_name] for r in results]
    eval_means = [r['eval_mean'] for r in results]
    eval_stds = [r['eval_std'] for r in results]

    axes[0].errorbar(range(len(params)), eval_means, yerr=eval_stds,
                     marker='o', capsize=5, linewidth=2, markersize=8)
    axes[0].set_xticks(range(len(params)))
    axes[0].set_xticklabels([str(p) for p in params])
    axes[0].set_xlabel(param_name.replace('_', ' ').title())
    axes[0].set_ylabel('Evaluation Score')
    axes[0].set_title(f'{param_name.replace("_", " ").title()} vs Performance')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Training curves
    for r, param in zip(results, params):
        scores = r['training_scores']
        window = 50
        if len(scores) >= window:
            smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
            axes[1].plot(smoothed, label=f'{param_name}={param}', linewidth=1.5)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Score (Smoothed)')
    axes[1].set_title('Training Progress')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Experiment: {exp_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{exp_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/{exp_name.lower().replace(' ', '_')}.png")


def run_all_experiments():
    """Run all hyperparameter experiments and generate report."""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING EXPERIMENTS")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs('results/experiments', exist_ok=True)
    all_results = {}

    # Experiment 1: Learning Rate
    lr_results = run_learning_rate_experiment()
    all_results['learning_rate'] = lr_results
    plot_experiment_results('Learning Rate Impact', lr_results, 'learning_rate')

    # Experiment 2: Discount Factor
    gamma_results = run_discount_factor_experiment()
    all_results['discount_factor'] = gamma_results
    plot_experiment_results('Discount Factor Impact', gamma_results, 'discount_factor')

    # Experiment 3: Exploration Decay
    eps_results = run_exploration_experiment()
    all_results['epsilon_decay'] = eps_results
    plot_experiment_results('Exploration Decay Impact', eps_results, 'epsilon_decay')

    # Experiment 4: Q-Learning vs SARSA
    comparison_results = run_tabular_comparison()
    all_results['tabular_comparison'] = comparison_results

    # Generate summary table
    summary_data = []

    # Best from each experiment
    best_lr = max(lr_results, key=lambda x: x['eval_mean'])
    summary_data.append({
        'Experiment': 'Learning Rate',
        'Best Value': best_lr['learning_rate'],
        'Eval Score': f"{best_lr['eval_mean']:.1f} +/- {best_lr['eval_std']:.1f}"
    })

    best_gamma = max(gamma_results, key=lambda x: x['eval_mean'])
    summary_data.append({
        'Experiment': 'Discount Factor',
        'Best Value': best_gamma['discount_factor'],
        'Eval Score': f"{best_gamma['eval_mean']:.1f} +/- {best_gamma['eval_std']:.1f}"
    })

    best_eps = max(eps_results, key=lambda x: x['eval_mean'])
    summary_data.append({
        'Experiment': 'Epsilon Decay',
        'Best Value': best_eps['epsilon_decay'],
        'Eval Score': f"{best_eps['eval_mean']:.1f} +/- {best_eps['eval_std']:.1f}"
    })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/experiments/hyperparameter_summary.csv', index=False)

    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70)

    # Save detailed results
    with open('results/experiments/detailed_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in all_results.items():
            if isinstance(value, list):
                serializable_results[key] = [
                    {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in r.items()}
                    for r in value
                ]
            elif isinstance(value, dict):
                serializable_results[key] = {
                    k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                        for kk, vv in v.items()}
                    for k, v in value.items()
                }
        json.dump(serializable_results, f, indent=2)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Results saved to results/experiments/")

    return all_results


if __name__ == "__main__":
    run_all_experiments()
