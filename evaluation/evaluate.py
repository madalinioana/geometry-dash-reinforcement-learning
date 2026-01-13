import numpy as np
from tqdm import tqdm

def evaluate_agent(agent, env, n_episodes=100, render=False):
    all_rewards = []
    all_scores = []
    all_lengths = []
    
    for episode in tqdm(range(n_episodes), desc="Evaluating"):
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
        
        all_rewards.append(episode_reward)
        all_scores.append(info.get('score', 0))
        all_lengths.append(episode_length)
    
    return {
        'mean_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'mean_score': float(np.mean(all_scores)),
        'std_score': float(np.std(all_scores)),
        'median_score': float(np.median(all_scores)),
        'max_score': int(np.max(all_scores)),
        'min_score': int(np.min(all_scores)),
        'mean_length': float(np.mean(all_lengths)),
        'std_length': float(np.std(all_lengths)),
        'scores': all_scores,
        'rewards': all_rewards,
        'lengths': all_lengths
    }
