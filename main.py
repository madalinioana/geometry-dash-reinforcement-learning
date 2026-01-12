"""
Reinforcement Learning Geometry Dash - Main Script
"""

import argparse
import os
import sys
import pygame  # Necesită import pentru detectarea tastelor
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ATENȚIE: Import corect din folderul environment
from environment.wrappers import FrameSkipWrapper, FrameStackWrapper, NormalizeObservation, RewardShapingWrapper

def ensure_directories():
    dirs = ['results/models', 'results/logs', 'results/plots', 'results/experiments']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def train_agent(agent_type):
    if agent_type == 'dqn':
        from training.train_dqn import train_dqn
        print("\n" + "="*60)
        print("TRAINING DQN AGENT (Precision Mode)")
        print("="*60)
        train_dqn()
    elif agent_type == 'ppo':
        from training.train_ppo import train_ppo
        train_ppo()
    elif agent_type == 'all':
        train_agent('dqn')
        train_agent('ppo')
    else:
        print(f"Agent {agent_type} not implemented in this quick-fix.")

def plot_results():
    from analysis.plot_results import plot_training_curves
    plot_training_curves()

def demo_agent(agent_type):
    from environment import ImpossibleGameEnv

    print(f"\n" + "="*60)
    print(f"DEMO: {agent_type.upper()} AGENT")
    print("="*60)

    env = ImpossibleGameEnv(render_mode="human", max_steps=10000)

    if agent_type == 'dqn':
        from agents.deep.dqn_agent import DQNAgent
        
        print("Applying DQN Wrappers (Skip=2, Stack=4)...")
        # FIX: Același skip ca la antrenament!
        env = FrameSkipWrapper(env, skip=2) 
        env = FrameStackWrapper(env, n_frames=4)
        env = NormalizeObservation(env)
        
        agent = DQNAgent(env.action_space, env.observation_space)
        model_path = 'results/models/dqn_agent_best.pth'
        if not os.path.exists(model_path):
             model_path = 'results/models/dqn_agent.pth'

    elif agent_type == 'ppo':
        from agents.policy.ppo_agent import PPOAgent
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        env.close() 
        env = DummyVecEnv([lambda: ImpossibleGameEnv(render_mode="human")])
        stats_path = 'results/models/vec_normalize.pkl'
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env)
            env.training = False 
            env.norm_reward = False
        agent = PPOAgent(env)
        model_path = 'results/models/ppo_agent.zip'
    else:
        print("Unknown agent")
        return

    try:
        if agent_type == 'ppo': agent.load(model_path)
        else: agent.load(model_path)
        print(f"Loaded: {model_path}")
    except:
        print("Model not found. Train first.")
        return

    try:
        episodes = 0
        while True:
            obs_data = env.reset()
            if agent_type == 'ppo': obs = obs_data
            elif isinstance(obs_data, tuple): obs = obs_data[0]
            else: obs = obs_data

            done = False
            total_reward = 0
            
            while not done:
                action = agent.select_action(obs, training=False)

                if agent_type == 'ppo':
                    obs, reward, done_array, info_array = env.step([action])
                    done = done_array[0]
                    reward = reward[0]
                    info = info_array[0]
                    env.render() 
                else:
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_result
                    env.render()

                total_reward += reward
            
            episodes += 1
            score = info.get('score', 0) if isinstance(info, dict) else 0
            print(f"Episode {episodes}: Score = {score}, Reward = {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\nDemo ended.")
    finally:
        env.close()

def play_game():
    """Permite utilizatorului să joace manual."""
    from environment import ImpossibleGameEnv
    
    print("\n" + "="*60)
    print("MANUAL PLAY MODE")
    print("Controls: SPACE, UP, or W to Jump")
    print("="*60)

    # Pentru joc manual NU folosim wrappers (vrem 60 FPS fluid, nu frame skip)
    env = ImpossibleGameEnv(render_mode="human", max_steps=10000)
    env.reset()

    try:
        running = True
        while running:
            # 1. Gestionare evenimente Pygame (inclusiv X de la fereastră)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 2. Input manual
            keys = pygame.key.get_pressed()
            action = 0
            if keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w]:
                action = 1
            
            # 3. Step
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            # 4. Reset la moarte
            if terminated or truncated:
                print(f"CRASH! Score: {info.get('score', 0)}")
                env.reset()
                
    except KeyboardInterrupt:
        print("\nPlay ended.")
    finally:
        env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='Train agent (dqn, ppo)')
    parser.add_argument('--demo', type=str, help='Demo agent (dqn, ppo)')
    parser.add_argument('--play', action='store_true', help='Play manually')
    parser.add_argument('--plots', action='store_true', help='Generate plots')
    args = parser.parse_args()

    ensure_directories()

    if args.play:
        play_game()
    elif args.train: 
        train_agent(args.train.lower())
    elif args.demo: 
        demo_agent(args.demo.lower())
    elif args.plots: 
        plot_results()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()