from stable_baselines3 import PPO
from agents.base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """Wrapper pentru Stable-Baselines3 PPO."""
    def __init__(self, env, **kwargs):
        super().__init__(env.action_space, env.observation_space)
        
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            **kwargs
        )
    
    def select_action(self, observation, training=True):
        action, _ = self.model.predict(observation, deterministic=not training)
        return action
    
    def update(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)
    
    def save(self, path):
        self.model.save(path)
        print(f"PPO agent saved to {path}")
    
    def load(self, path):
        self.model = PPO.load(path)
        print(f"PPO agent loaded from {path}")