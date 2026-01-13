import numpy as np
import pandas as pd
import os
import gymnasium as gym
from tqdm import tqdm
from environment.geometry_dash_env import ImpossibleGameEnv
from environment.wrappers import FrameSkipWrapper, NormalizeObservation
from agents.deep.dqn_agent import DQNAgent
from agents.tabular.q_learning_agent import QLearningAgent
from agents.tabular.sarsa_agent import SARSAAgent
from agents.policy.ppo_agent import PPOAgent

def extended_evaluation(n_episodes=500):
    env_dqn = ImpossibleGameEnv(render_mode=None)
    env_dqn = FrameSkipWrapper(env_dqn, skip=2)
    env_dqn = gym.wrappers.FrameStackObservation(env_dqn, stack_size=4)
    env_dqn = gym.wrappers.FlattenObservation(env_dqn)
    env_dqn = NormalizeObservation(env_dqn)
    
    env_tabular = ImpossibleGameEnv(render_mode=None)
    env_tabular = FrameSkipWrapper(env_tabular, skip=4)
    env_tabular = gym.wrappers.FlattenObservation(env_tabular)
    env_tabular = NormalizeObservation(env_tabular)
    
    agents_config = [
        ('DQN', DQNAgent, 'results/models/dqn_agent_best.pth', env_dqn),
        ('Q-Learning', QLearningAgent, 'results/models/q_learning_agent.pkl', env_tabular),
        ('SARSA', SARSAAgent, 'results/models/sarsa_agent.pkl', env_tabular),
        ('PPO', PPOAgent, 'results/models/ppo_agent.zip', env_tabular)
    ]
    
    all_results = {}
    
    for name, agent_class, model_path, env in agents_config:
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        
        print(f"\nEvaluating {name} for {n_episodes} episodes...")
        
        if name == 'PPO':
            agent = agent_class(env)
        else:
            agent = agent_class(env.action_space, env.observation_space)
        
        agent.load(model_path)
        
        scores = []
        rewards = []
        lengths = []
        
        for episode in tqdm(range(n_episodes), desc=f"{name}"):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            scores.append(info.get('score', 0))
            rewards.append(episode_reward)
            lengths.append(episode_length)
        
        all_results[name] = {
            'scores': scores,
            'rewards': rewards,
            'lengths': lengths,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'median_score': np.median(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards)
        }
        
        print(f"{name} - Mean: {np.mean(scores):.2f}, Std: {np.std(scores):.2f}, Max: {np.max(scores)}, Min: {np.min(scores)}")
    
    env_dqn.close()
    env_tabular.close()
    
    os.makedirs('results', exist_ok=True)
    
    summary_data = []
    for name, results in all_results.items():
        summary_data.append({
            'Agent': name,
            'Mean Score': f"{results['mean_score']:.2f}",
            'Std Score': f"{results['std_score']:.2f}",
            'Median Score': f"{results['median_score']:.2f}",
            'Max Score': int(results['max_score']),
            'Min Score': int(results['min_score']),
            'Mean Reward': f"{results['mean_reward']:.2f}",
            'Std Reward': f"{results['std_reward']:.2f}"
        })
    
    df = pd.DataFrame(summary_data)
    print("\nExtended Evaluation Results:")
    print(df.to_string(index=False))
    df.to_csv('results/extended_evaluation.csv', index=False)
    
    for name, results in all_results.items():
        np.save(f"results/logs/{name.lower().replace('-', '_')}_eval_scores.npy", results['scores'])
        np.save(f"results/logs/{name.lower().replace('-', '_')}_eval_rewards.npy", results['rewards'])
    
    print("\nData saved to results/extended_evaluation.csv and results/logs/")
    return all_results

if __name__ == '__main__':
    results = extended_evaluation(n_episodes=500)
