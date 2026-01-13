import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

class TrainingVisualizer:
    def __init__(self, output_dir='results/plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def smooth_curve(self, data, window=50):
        if len(data) < window:
            return data
        return uniform_filter1d(data, size=window, mode='nearest')
    
    def plot_learning_curves(self, agents_stats, metric='scores'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(agents_stats)))
        
        for agent, color in zip(agents_stats, colors):
            data = agent.scores if metric == 'scores' else agent.rewards
            episodes = np.arange(len(data))
            smoothed = self.smooth_curve(data, window=100)
            
            ax1.plot(episodes, smoothed, label=agent.name, color=color, linewidth=2, alpha=0.9)
            ax1.fill_between(episodes, 
                            self.smooth_curve(data, 50), 
                            self.smooth_curve(data, 200),
                            color=color, alpha=0.1)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score' if metric == 'scores' else 'Reward')
        ax1.set_title(f'Learning Curves - {"Scores" if metric == "scores" else "Rewards"}')
        ax1.legend(frameon=True, loc='best')
        ax1.grid(True, alpha=0.3)
        
        for agent, color in zip(agents_stats, colors):
            data = agent.scores if metric == 'scores' else agent.rewards
            if len(data) >= 1000:
                windows = [100, 500, 1000]
                for window in windows:
                    rolling = pd.Series(data).rolling(window).mean()
                    ax2.plot(rolling.index, rolling.values, 
                            label=f'{agent.name} (w={window})', 
                            color=color, linewidth=1.5, 
                            alpha=0.7 if window != 500 else 1.0,
                            linestyle='-' if window == 500 else '--')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Rolling Mean Score' if metric == 'scores' else 'Rolling Mean Reward')
        ax2.set_title('Convergence Analysis (Multiple Window Sizes)')
        ax2.legend(frameon=True, loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/learning_curves_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_distribution_comparison(self, agents_stats):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        data_for_violin = []
        labels = []
        for agent in agents_stats:
            data_for_violin.append(agent.scores)
            labels.append(agent.name)
        
        parts = ax1.violinplot(data_for_violin, positions=range(len(agents_stats)), 
                              showmeans=True, showmedians=True)
        ax1.set_xticks(range(len(agents_stats)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Score Distribution')
        ax1.set_title('Score Distributions (Violin Plot)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(agents_stats)))
        for agent, color in zip(agents_stats, colors):
            ax2.hist(agent.scores, bins=50, alpha=0.5, label=agent.name, 
                    color=color, density=True)
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Density')
        ax2.set_title('Score Histograms (Overlayed)')
        ax2.legend(frameon=True)
        ax2.grid(True, alpha=0.3)
        
        bp = ax3.boxplot(data_for_violin, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.set_ylabel('Score')
        ax3.set_title('Box Plot Comparison')
        ax3.grid(True, alpha=0.3, axis='y')
        
        stats_data = []
        for agent in agents_stats:
            final_100 = agent.scores[-100:] if len(agent.scores) >= 100 else agent.scores
            stats_data.append({
                'Agent': agent.name,
                'Mean': np.mean(agent.scores),
                'Final 100': np.mean(final_100),
                'Max': np.max(agent.scores),
                'Std': np.std(agent.scores)
            })
        df = pd.DataFrame(stats_data)
        
        x = np.arange(len(agents_stats))
        width = 0.25
        
        ax4.bar(x - width, df['Mean'], width, label='Overall Mean', alpha=0.8)
        ax4.bar(x, df['Final 100'], width, label='Final 100 Eps', alpha=0.8)
        ax4.bar(x + width, df['Max'], width, label='Max', alpha=0.8)
        
        ax4.set_xlabel('Agent')
        ax4.set_ylabel('Score')
        ax4.set_title('Performance Metrics Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.legend(frameon=True)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_heatmap(self, agents_stats):
        metrics = []
        for agent in agents_stats:
            basic = agent.compute_basic_stats()
            conv = agent.compute_convergence_metrics()
            eff = agent.compute_learning_efficiency()
            
            metrics.append({
                'Mean': basic['mean'],
                'Median': basic['median'],
                'Max': basic['max'],
                'Stability': -conv.get('stability', 0),
                'Improvement': conv.get('improvement', 0),
                'Sample Eff': eff['sample_efficiency']
            })
        
        df = pd.DataFrame(metrics, index=[a.name for a in agents_stats])
        
        df_normalized = (df - df.min()) / (df.max() - df.min())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(df, annot=True, fmt='.2f', cmap='YlGnBu', 
                   cbar_kws={'label': 'Raw Value'}, ax=ax1, linewidths=0.5)
        ax1.set_title('Performance Metrics Heatmap (Raw Values)')
        ax1.set_ylabel('Agent')
        
        sns.heatmap(df_normalized, annot=True, fmt='.2f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Normalized Score'}, ax=ax2, linewidths=0.5)
        ax2.set_title('Performance Metrics Heatmap (Normalized)')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_radar_chart(self, agents_stats):
        metrics_names = ['Mean\nScore', 'Max\nScore', 'Stability', 
                        'Improvement', 'Sample\nEfficiency', 'Final\nPerf']
        
        num_vars = len(metrics_names)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(agents_stats)))
        
        for agent, color in zip(agents_stats, colors):
            basic = agent.compute_basic_stats()
            conv = agent.compute_convergence_metrics()
            eff = agent.compute_learning_efficiency()
            
            values = [
                basic['mean'],
                basic['max'],
                -conv.get('stability', 0),
                conv.get('improvement', 0),
                eff['sample_efficiency'],
                conv.get('final_performance', basic['mean'])
            ]
            
            max_vals = [max(a.compute_basic_stats()['mean'] for a in agents_stats),
                       max(a.compute_basic_stats()['max'] for a in agents_stats),
                       max(-a.compute_convergence_metrics().get('stability', 0) for a in agents_stats),
                       max(a.compute_convergence_metrics().get('improvement', 0) for a in agents_stats),
                       max(a.compute_learning_efficiency()['sample_efficiency'] for a in agents_stats),
                       max(a.compute_convergence_metrics().get('final_performance', 0) for a in agents_stats)]
            
            normalized = [v / m if m > 0 else 0 for v, m in zip(values, max_vals)]
            normalized += normalized[:1]
            
            ax.plot(angles, normalized, 'o-', linewidth=2, label=agent.name, color=color)
            ax.fill(angles, normalized, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.set_title('Multi-Metric Performance Comparison (Radar Chart)', 
                    size=14, weight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_convergence_analysis(self, agents_stats):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(agents_stats)))
        
        for agent, color in zip(agents_stats, colors):
            window_sizes = [50, 100, 200, 500]
            stds = []
            for w in window_sizes:
                if len(agent.scores) >= w:
                    rolling = pd.Series(agent.scores).rolling(w).mean()
                    stds.append(np.std(rolling.dropna()))
                else:
                    stds.append(np.nan)
            
            ax1.plot(window_sizes, stds, 'o-', label=agent.name, 
                    color=color, linewidth=2, markersize=8)
        
        ax1.set_xlabel('Window Size')
        ax1.set_ylabel('Std of Rolling Mean')
        ax1.set_title('Stability vs Window Size')
        ax1.legend(frameon=True)
        ax1.grid(True, alpha=0.3)
        
        for agent, color in zip(agents_stats, colors):
            if len(agent.scores) >= 100:
                segment_size = len(agent.scores) // 10
                means = [np.mean(agent.scores[i*segment_size:(i+1)*segment_size]) 
                        for i in range(10)]
                ax2.plot(range(1, 11), means, 'o-', label=agent.name, 
                        color=color, linewidth=2, markersize=8)
        
        ax2.set_xlabel('Training Phase (Decile)')
        ax2.set_ylabel('Average Score')
        ax2.set_title('Performance Across Training Phases')
        ax2.legend(frameon=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(1, 11))
        
        for agent, color in zip(agents_stats, colors):
            cumulative = np.cumsum(agent.rewards)
            episodes = np.arange(len(cumulative))
            ax3.plot(episodes, cumulative, label=agent.name, 
                    color=color, linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Cumulative Reward')
        ax3.set_title('Sample Efficiency (Cumulative Rewards)')
        ax3.legend(frameon=True)
        ax3.grid(True, alpha=0.3)
        
        improvement_data = []
        for agent in agents_stats:
            if len(agent.scores) >= 200:
                initial = np.mean(agent.scores[:100])
                final = np.mean(agent.scores[-100:])
                improvement = ((final - initial) / max(abs(initial), 1)) * 100
                improvement_data.append({
                    'Agent': agent.name,
                    'Initial': initial,
                    'Final': final,
                    'Improvement': improvement
                })
        
        if improvement_data:
            df = pd.DataFrame(improvement_data)
            x = np.arange(len(df))
            width = 0.35
            
            ax4.bar(x - width/2, df['Initial'], width, label='Initial (first 100)', alpha=0.8)
            ax4.bar(x + width/2, df['Final'], width, label='Final (last 100)', alpha=0.8)
            
            for i, row in df.iterrows():
                ax4.text(i, max(row['Initial'], row['Final']) + 1, 
                        f"+{row['Improvement']:.1f}%", 
                        ha='center', va='bottom', fontsize=9, weight='bold')
            
            ax4.set_xlabel('Agent')
            ax4.set_ylabel('Average Score')
            ax4.set_title('Learning Progress (Initial vs Final)')
            ax4.set_xticks(x)
            ax4.set_xticklabels(df['Agent'], rotation=45, ha='right')
            ax4.legend(frameon=True)
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_plots(self, agents_stats):
        self.plot_learning_curves(agents_stats, metric='scores')
        self.plot_learning_curves(agents_stats, metric='rewards')
        self.plot_distribution_comparison(agents_stats)
        self.plot_performance_heatmap(agents_stats)
        self.plot_radar_chart(agents_stats)
        self.plot_convergence_analysis(agents_stats)
