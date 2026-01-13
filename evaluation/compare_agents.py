from analysis.statistics import AgentStatistics, MultiAgentComparison, generate_full_report
from analysis.visualizations import TrainingVisualizer
from evaluation.evaluate import evaluate_agent
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.geometry_dash_env import GeometryDashEnv
from environment.wrappers import FrameSkipWrapper, NormalizeObservationWrapper
from agents.deep.dqn_agent import DQNAgent
from agents.tabular.q_learning_agent import QLearningAgent
from agents.tabular.sarsa_agent import SARSAAgent
from agents.policy.ppo_agent import PPOAgent

def compare_all_agents(n_episodes=100, render=False):
    env = GeometryDashEnv(render_mode='human' if render else None)
    env = FrameSkipWrapper(env, skip=4)
    env = NormalizeObservationWrapper(env)
    
    agents_config = [
        ('DQN', DQNAgent, 'results/models/dqn_agent_best.pth'),
        ('Q-Learning', QLearningAgent, 'results/models/q_learning_agent.pkl'),
        ('SARSA', SARSAAgent, 'results/models/sarsa_agent.pkl'),
        ('PPO', PPOAgent, 'results/models/ppo_agent.zip')
    ]
    
    results = []
    
    for name, agent_class, model_path in agents_config:
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        
        print(f"Evaluating {name}...")
        agent = agent_class(env.observation_space, env.action_space)
        agent.load(model_path)
        
        metrics = evaluate_agent(agent, env, n_episodes=n_episodes, render=render)
        
        results.append({
            'Agent': name,
            'Mean Score': f"{metrics['mean_score']:.2f}",
            'Std Score': f"{metrics['std_score']:.2f}",
            'Median Score': f"{metrics['median_score']:.2f}",
            'Max Score': int(metrics['max_score']),
            'Min Score': int(metrics['min_score']),
            'Mean Reward': f"{metrics['mean_reward']:.2f}",
            'Success Rate': f"{metrics.get('success_rate', 0)*100:.1f}%"
        })
        
        print(f"Mean: {metrics['mean_score']:.2f}, Max: {metrics['max_score']}")
    
    env.close()
    
    df = pd.DataFrame(results)
    
    print("\nAgent Comparison Results:")
    print(df.to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/evaluation_comparison.csv', index=False)
    print("Results saved to results/evaluation_comparison.csv")
    
    return df

if __name__ == '__main__':
    compare_all_agents(n_episodes=100, render=False)

if __name__ == '__main__':
    compare_all_agents(n_episodes=100, render=False)
