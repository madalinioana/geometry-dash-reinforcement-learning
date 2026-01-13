import numpy as np
import pandas as pd
from scipy import stats
import json
import os

class AgentStatistics:
    def __init__(self, agent_name, rewards, scores, lengths=None):
        self.name = agent_name
        self.rewards = np.array(rewards)
        self.scores = np.array(scores)
        self.lengths = np.array(lengths) if lengths is not None else None
        
    def compute_basic_stats(self):
        return {
            'mean': float(np.mean(self.scores)),
            'std': float(np.std(self.scores)),
            'min': float(np.min(self.scores)),
            'max': float(np.max(self.scores)),
            'median': float(np.median(self.scores)),
            'q25': float(np.percentile(self.scores, 25)),
            'q75': float(np.percentile(self.scores, 75)),
            'iqr': float(np.percentile(self.scores, 75) - np.percentile(self.scores, 25))
        }
    
    def compute_convergence_metrics(self, window=100):
        if len(self.scores) < window:
            return {}
        
        final_performance = np.mean(self.scores[-window:])
        initial_performance = np.mean(self.scores[:window])
        improvement = final_performance - initial_performance
        
        rolling_mean = pd.Series(self.scores).rolling(window).mean()
        stability = np.std(rolling_mean[-window:])
        
        return {
            'final_performance': float(final_performance),
            'initial_performance': float(initial_performance),
            'improvement': float(improvement),
            'stability': float(stability),
            'convergence_rate': float(improvement / len(self.scores))
        }
    
    def compute_learning_efficiency(self):
        cumsum_rewards = np.cumsum(self.rewards)
        total_reward = cumsum_rewards[-1]
        episodes_to_50pct = np.argmax(cumsum_rewards >= total_reward * 0.5) + 1
        episodes_to_90pct = np.argmax(cumsum_rewards >= total_reward * 0.9) + 1
        
        return {
            'total_reward': float(total_reward),
            'episodes_to_50pct_reward': int(episodes_to_50pct),
            'episodes_to_90pct_reward': int(episodes_to_90pct),
            'sample_efficiency': float(total_reward / len(self.rewards))
        }
    
    def compute_all(self):
        stats = {
            'agent': self.name,
            'episodes': len(self.scores),
            'basic_stats': self.compute_basic_stats(),
            'convergence': self.compute_convergence_metrics(),
            'efficiency': self.compute_learning_efficiency()
        }
        return stats

class MultiAgentComparison:
    def __init__(self, agents_stats):
        self.agents = agents_stats
        
    def compare_performance(self):
        comparison = []
        for agent in self.agents:
            stats = agent.compute_basic_stats()
            comparison.append({
                'Agent': agent.name,
                'Mean Score': f"{stats['mean']:.2f}",
                'Std': f"{stats['std']:.2f}",
                'Median': f"{stats['median']:.2f}",
                'Max': int(stats['max']),
                'Q25-Q75': f"{stats['q25']:.0f}-{stats['q75']:.0f}"
            })
        return pd.DataFrame(comparison)
    
    def statistical_tests(self):
        if len(self.agents) < 2:
            return {}
        
        results = {}
        agent_pairs = [(self.agents[i], self.agents[j]) 
                      for i in range(len(self.agents)) 
                      for j in range(i+1, len(self.agents))]
        
        for a1, a2 in agent_pairs:
            t_stat, p_value = stats.ttest_ind(a1.scores, a2.scores)
            u_stat, p_value_mw = stats.mannwhitneyu(a1.scores, a2.scores, alternative='two-sided')
            
            results[f"{a1.name}_vs_{a2.name}"] = {
                't_test_pvalue': float(p_value),
                'mann_whitney_pvalue': float(p_value_mw),
                'significantly_different': bool(p_value < 0.05),
                'mean_diff': float(np.mean(a1.scores) - np.mean(a2.scores))
            }
        
        return results
    
    def rank_agents(self):
        rankings = []
        for agent in self.agents:
            mean_score = np.mean(agent.scores)
            stability = -np.std(agent.scores)
            final_perf = np.mean(agent.scores[-100:]) if len(agent.scores) >= 100 else mean_score
            
            composite_score = (mean_score * 0.5) + (final_perf * 0.3) + (stability * 0.2)
            
            rankings.append({
                'agent': agent.name,
                'mean_score': float(mean_score),
                'final_performance': float(final_perf),
                'stability': float(-stability),
                'composite_score': float(composite_score)
            })
        
        rankings.sort(key=lambda x: x['composite_score'], reverse=True)
        for i, r in enumerate(rankings):
            r['rank'] = i + 1
        
        return rankings

def generate_full_report(agents_stats, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        'agents': {},
        'comparison': {},
        'rankings': []
    }
    
    for agent_stat in agents_stats:
        report['agents'][agent_stat.name] = agent_stat.compute_all()
    
    comparator = MultiAgentComparison(agents_stats)
    
    comparison_df = comparator.compare_performance()
    comparison_df.to_csv(f'{output_dir}/comparison_summary.csv', index=False)
    report['comparison']['table'] = comparison_df.to_dict('records')
    
    report['comparison']['statistical_tests'] = comparator.statistical_tests()
    report['rankings'] = comparator.rank_agents()
    
    with open(f'{output_dir}/analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    rankings_df = pd.DataFrame(report['rankings'])
    rankings_df.to_csv(f'{output_dir}/agent_rankings.csv', index=False)
    
    return report

def load_training_data(agent_name, base_path='results/logs'):
    rewards_file = f'{base_path}/{agent_name}_rewards.npy'
    scores_file = f'{base_path}/{agent_name}_scores.npy'
    
    if os.path.exists(rewards_file) and os.path.exists(scores_file):
        rewards = np.load(rewards_file)
        scores = np.load(scores_file)
        return AgentStatistics(agent_name.upper().replace('_', '-'), rewards, scores)
    return None
