import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("darkgrid")


def smooth_curve(data, window=50):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_training_curves():
    """Generate comprehensive training curve visualizations."""

    os.makedirs('results/plots', exist_ok=True)

    # Define agent configurations
    agents_config = {
        'Q-Learning': {
            'rewards_file': 'results/logs/q_learning_rewards.npy',
            'scores_file': 'results/logs/q_learning_scores.npy',
            'color': '#3498db',
            'exists': False
        },
        'SARSA': {
            'rewards_file': 'results/logs/sarsa_rewards.npy',
            'scores_file': 'results/logs/sarsa_scores.npy',
            'color': '#2ecc71',
            'exists': False
        },
        'DQN': {
            'rewards_file': 'results/logs/dqn_rewards.npy',
            'scores_file': 'results/logs/dqn_scores.npy',
            'color': '#e74c3c',
            'exists': False
        }
    }

    # Load available data
    for name, config in agents_config.items():
        if os.path.exists(config['rewards_file']) and os.path.exists(config['scores_file']):
            config['rewards'] = np.load(config['rewards_file'])
            config['scores'] = np.load(config['scores_file'])
            config['exists'] = True
            print(f"Loaded {name} data: {len(config['rewards'])} episodes")

    available_agents = [name for name, config in agents_config.items() if config['exists']]

    if not available_agents:
        print("No training logs found! Train agents first.")
        return

    # Figure 1: Individual training curves (2 rows: rewards, scores)
    n_agents = len(available_agents)
    fig1, axes1 = plt.subplots(2, n_agents, figsize=(5*n_agents, 8))

    if n_agents == 1:
        axes1 = axes1.reshape(2, 1)

    for i, name in enumerate(available_agents):
        config = agents_config[name]
        window = 50

        # Rewards
        axes1[0, i].plot(config['rewards'], alpha=0.3, color=config['color'])
        if len(config['rewards']) >= window:
            smoothed = smooth_curve(config['rewards'], window)
            axes1[0, i].plot(range(window-1, len(config['rewards'])), smoothed,
                            color=config['color'], linewidth=2, label='Smoothed')
        axes1[0, i].set_title(f'{name}: Rewards', fontsize=12, fontweight='bold')
        axes1[0, i].set_xlabel('Episode')
        axes1[0, i].set_ylabel('Reward')
        axes1[0, i].legend()

        # Scores
        axes1[1, i].plot(config['scores'], alpha=0.3, color=config['color'])
        if len(config['scores']) >= window:
            smoothed = smooth_curve(config['scores'], window)
            axes1[1, i].plot(range(window-1, len(config['scores'])), smoothed,
                            color=config['color'], linewidth=2, label='Smoothed')
        axes1[1, i].set_title(f'{name}: Scores', fontsize=12, fontweight='bold')
        axes1[1, i].set_xlabel('Episode')
        axes1[1, i].set_ylabel('Score')
        axes1[1, i].legend()

    plt.suptitle('Training Progress by Agent', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/training_curves_individual.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/training_curves_individual.png")

    # Figure 2: Comparison plots
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Score comparison (all agents on same plot)
    for name in available_agents:
        config = agents_config[name]
        if len(config['scores']) >= 50:
            smoothed = smooth_curve(config['scores'], 50)
            axes2[0, 0].plot(smoothed, label=name, color=config['color'], linewidth=2)
    axes2[0, 0].set_title('Score Learning Curves Comparison', fontsize=12, fontweight='bold')
    axes2[0, 0].set_xlabel('Episode')
    axes2[0, 0].set_ylabel('Score (Smoothed)')
    axes2[0, 0].legend()
    axes2[0, 0].grid(True, alpha=0.3)

    # Plot 2: Reward comparison
    for name in available_agents:
        config = agents_config[name]
        if len(config['rewards']) >= 50:
            smoothed = smooth_curve(config['rewards'], 50)
            axes2[0, 1].plot(smoothed, label=name, color=config['color'], linewidth=2)
    axes2[0, 1].set_title('Reward Learning Curves Comparison', fontsize=12, fontweight='bold')
    axes2[0, 1].set_xlabel('Episode')
    axes2[0, 1].set_ylabel('Reward (Smoothed)')
    axes2[0, 1].legend()
    axes2[0, 1].grid(True, alpha=0.3)

    # Plot 3: Score distribution histogram
    for name in available_agents:
        config = agents_config[name]
        axes2[1, 0].hist(config['scores'], bins=30, alpha=0.5,
                        label=name, color=config['color'])
    axes2[1, 0].set_title('Score Distribution', fontsize=12, fontweight='bold')
    axes2[1, 0].set_xlabel('Score')
    axes2[1, 0].set_ylabel('Frequency')
    axes2[1, 0].legend()

    # Plot 4: Final performance summary
    final_scores = []
    final_rewards = []
    agent_names = []
    colors = []

    for name in available_agents:
        config = agents_config[name]
        last_100_scores = config['scores'][-100:] if len(config['scores']) >= 100 else config['scores']
        last_100_rewards = config['rewards'][-100:] if len(config['rewards']) >= 100 else config['rewards']
        final_scores.append(np.mean(last_100_scores))
        final_rewards.append(np.mean(last_100_rewards))
        agent_names.append(name)
        colors.append(config['color'])

    x = np.arange(len(agent_names))
    width = 0.35

    bars1 = axes2[1, 1].bar(x - width/2, final_scores, width, label='Avg Score (last 100)',
                            color=colors, alpha=0.8)
    axes2[1, 1].set_ylabel('Score', color='blue')
    axes2[1, 1].tick_params(axis='y', labelcolor='blue')

    ax2_twin = axes2[1, 1].twinx()
    bars2 = ax2_twin.bar(x + width/2, final_rewards, width, label='Avg Reward (last 100)',
                         color=colors, alpha=0.4, hatch='//')
    ax2_twin.set_ylabel('Reward', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')

    axes2[1, 1].set_title('Final Performance (Last 100 Episodes)', fontsize=12, fontweight='bold')
    axes2[1, 1].set_xticks(x)
    axes2[1, 1].set_xticklabels(agent_names)
    axes2[1, 1].legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    plt.suptitle('Reinforcement Learning Training Analysis\nGeometry Dash Environment',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/training_curves_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/training_curves_comparison.png")

    # Figure 3: Convergence analysis
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

    # Rolling mean with different windows
    windows = [10, 50, 100]
    for name in available_agents:
        config = agents_config[name]
        for w in windows:
            if len(config['scores']) >= w:
                smoothed = smooth_curve(config['scores'], w)
                alpha = 0.3 + 0.2 * windows.index(w)
                if w == 50:
                    axes3[0].plot(smoothed, label=f'{name}', color=config['color'],
                                 linewidth=2, alpha=1.0)

    axes3[0].set_title('Score Convergence Analysis', fontsize=12, fontweight='bold')
    axes3[0].set_xlabel('Episode')
    axes3[0].set_ylabel('Score (50-episode moving avg)')
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)

    # Variance over time
    variance_window = 100
    for name in available_agents:
        config = agents_config[name]
        if len(config['scores']) >= variance_window:
            variances = []
            for i in range(variance_window, len(config['scores'])):
                variances.append(np.std(config['scores'][i-variance_window:i]))
            axes3[1].plot(variances, label=name, color=config['color'], linewidth=2)

    axes3[1].set_title('Score Stability (Rolling Std Dev)', fontsize=12, fontweight='bold')
    axes3[1].set_xlabel('Episode')
    axes3[1].set_ylabel(f'Std Dev ({variance_window}-episode window)')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/plots/convergence_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/convergence_analysis.png")

    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)
    for name in available_agents:
        config = agents_config[name]
        print(f"\n{name}:")
        print(f"  Episodes: {len(config['scores'])}")
        print(f"  Mean Score: {np.mean(config['scores']):.2f} +/- {np.std(config['scores']):.2f}")
        print(f"  Max Score: {np.max(config['scores'])}")
        print(f"  Final Avg (last 100): {np.mean(config['scores'][-100:]):.2f}")
        print(f"  Mean Reward: {np.mean(config['rewards']):.2f}")
    print("="*60)


if __name__ == "__main__":
    plot_training_curves()
