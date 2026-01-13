import argparse
import os
import sys
import pygame

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.wrappers import FrameStackWrapper, NormalizeObservation

def ensure_directories():
    for d in ['results/models', 'results/logs', 'results/plots']:
        os.makedirs(d, exist_ok=True)

def train_agent(agent_type):
    if agent_type == 'all':
        for a in ['q_learning', 'sarsa', 'dqn', 'ppo']:
            train_agent(a)
        return
    
    agents = {
        'dqn': ('training.train_dqn', 'train_dqn'),
        'ppo': ('training.train_ppo', 'train_ppo'),
        'q_learning': ('training.train_q_learning', 'train_q_learning'),
        'sarsa': ('training.train_sarsa', 'train_sarsa')
    }
    
    if agent_type in agents:
        module_name, func_name = agents[agent_type]
        module = __import__(module_name, fromlist=[func_name])
        getattr(module, func_name)()
    else:
        print(f"Agent {agent_type} not implemented yet.")

def plot_results():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analysis.statistics import load_training_data, generate_full_report
    from analysis.visualizations import TrainingVisualizer
    
    agents = ['dqn', 'q_learning', 'sarsa', 'ppo']
    agents_stats = [a for a in [load_training_data(name) for name in agents] if a]
    
    if not agents_stats:
        print("No training data found")
        return
    
    print(f"Generating plots for {len(agents_stats)} agents...")
    generate_full_report(agents_stats, output_dir='results')
    visualizer = TrainingVisualizer(output_dir='results/plots')
    visualizer.generate_all_plots(agents_stats)
    print("Plots saved in results/plots/")


def demo_agent(agent_type):
    from environment import ImpossibleGameEnv

    print(f"Demo {agent_type}...")

    env = ImpossibleGameEnv(render_mode="human", max_steps=10000)

    if agent_type == 'dqn':
        from agents.deep.dqn_agent import DQNAgent
        
        env = FrameStackWrapper(env, n_frames=4)
        env = NormalizeObservation(env)
        
        agent = DQNAgent(env.action_space, env.observation_space)
        model_path = 'results/models/dqn_agent_best.pth'
        if not os.path.exists(model_path):
             model_path = 'results/models/dqn_agent.pth'

    elif agent_type == 'q_learning':
        from agents.tabular.q_learning_agent import QLearningAgent
        print("Loading Q-Learning Agent (no wrappers needed)...")
        agent = QLearningAgent(env.action_space, env.observation_space)
        model_path = 'results/models/q_learning_agent.pkl'

    elif agent_type == 'sarsa':
        from agents.tabular.sarsa_agent import SARSAAgent
        print("Loading SARSA Agent (no wrappers needed)...")
        agent = SARSAAgent(env.action_space, env.observation_space)
        model_path = 'results/models/sarsa_agent.pkl'

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
        agent.load(model_path)
    except:
        print("Model not found.")
        return

    try:
        episodes = 0
        while True:
            obs_data = env.reset()
            obs = obs_data if agent_type == 'ppo' else (obs_data[0] if isinstance(obs_data, tuple) else obs_data)
            done = False
            total_reward = 0
            
            while not done:
                action = agent.select_action(obs, training=False)

                if agent_type == 'ppo':
                    obs, reward, done_array, info_array = env.step([action])
                    done, reward, info = done_array[0], reward[0], info_array[0]
                else:
                    step_result = env.step(action)
                    obs, reward, terminated, truncated, info = step_result if len(step_result) == 5 else (*step_result[:3], False, step_result[3])
                    done = terminated or truncated if len(step_result) == 5 else step_result[2]
                
                env.render()
                total_reward += reward
            
            episodes += 1
            score = info.get('score', 0) if isinstance(info, dict) else 0
            print(f"Episode {episodes}: Score={score}, Reward={total_reward:.1f}")

    except KeyboardInterrupt:
        print("Done.")
    finally:
        env.close()

def play_game():
    from environment import ImpossibleGameEnv
    
    print("Manual play. SPACE/UP/W to jump.")

    env = ImpossibleGameEnv(render_mode="human", max_steps=10000)
    env.reset()

    try:
        running = True
        while running:
            if any(event.type == pygame.QUIT for event in pygame.event.get()):
                running = False
                continue

            keys = pygame.key.get_pressed()
            action = 1 if any([keys[pygame.K_SPACE], keys[pygame.K_UP], keys[pygame.K_w]]) else 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                print(f"Score: {info.get('score', 0)}")
                env.reset()
                
    except KeyboardInterrupt:
        print("Done.")
    finally:
        env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='Train agent (q_learning, sarsa, dqn, ppo, all)')
    parser.add_argument('--demo', type=str, help='Demo agent (q_learning, sarsa, dqn, ppo)')
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