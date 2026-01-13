import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.statistics import load_training_data, generate_full_report
from analysis.visualizations import TrainingVisualizer

def quick_analysis():
    agents = ['dqn', 'q_learning', 'sarsa', 'ppo']
    agents_stats = []
    
    for agent_name in agents:
        agent_stat = load_training_data(agent_name)
        if agent_stat:
            agents_stats.append(agent_stat)
    
    if not agents_stats:
        print("No training data found")
        return
    
    print(f"Generating plots for {len(agents_stats)} agents...")
    
    generate_full_report(agents_stats, output_dir='results')
    visualizer = TrainingVisualizer(output_dir='results/plots')
    visualizer.generate_all_plots(agents_stats)
    
    print("Done. Plots saved in results/plots/")

if __name__ == '__main__':
    quick_analysis()
