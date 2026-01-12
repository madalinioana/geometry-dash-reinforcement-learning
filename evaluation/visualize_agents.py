import gymnasium as gym
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ImpossibleGameEnv
from agents.tabular.q_learning_agent import QLearningAgent
from agents.tabular.sarsa_agent import SARSAAgent
from agents.deep.dqn_agent import DQNAgent
from agents.policy.ppo_agent import PPOAgent


def visualize_agent(agent_type, model_path=None, episodes=5):
    """Visualize a trained agent playing the game."""

    env = ImpossibleGameEnv(render_mode="human")

    # Default model paths
    default_paths = {
        'q_learning': 'results/models/q_learning_agent.pkl',
        'sarsa': 'results/models/sarsa_agent.pkl',
        'dqn': 'results/models/dqn_agent.pth',
        'ppo': 'results/models/ppo_agent'
    }

    if model_path is None:
        model_path = default_paths.get(agent_type)

    # Load agent based on type
    if agent_type == "q_learning":
        agent = QLearningAgent(env.action_space, env.observation_space)
        agent.load(model_path)
    elif agent_type == "sarsa":
        agent = SARSAAgent(env.action_space, env.observation_space)
        agent.load(model_path)
    elif agent_type == "dqn":
        agent = DQNAgent(env.action_space, env.observation_space)
        agent.load(model_path)
    elif agent_type == "ppo":
        agent = PPOAgent(env)
        agent.load(model_path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    print(f"\nVisualizing {agent_type.upper()} agent")
    print(f"Model: {model_path}")
    print("-" * 40)

    total_scores = []

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            env.render()

        total_scores.append(info['score'])
        print(f"Episode {episode+1}: Score = {info['score']}, Reward = {total_reward:.0f}")

    env.close()

    print("-" * 40)
    print(f"Average Score: {sum(total_scores)/len(total_scores):.1f}")
    print(f"Max Score: {max(total_scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize trained RL agent')
    parser.add_argument('--agent', type=str, required=True,
                        choices=['q_learning', 'sarsa', 'dqn', 'ppo'],
                        help='Agent type to visualize')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (optional, uses default if not specified)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to visualize')

    args = parser.parse_args()

    visualize_agent(args.agent, args.model, args.episodes)
