from gymnasium.envs.registration import register
from environment.geometry_dash_env import ImpossibleGameEnv
from environment.wrappers import FrameSkipWrapper, NormalizeObservation


register(
    id='GeometryDash-v0',
    entry_point='environment.geometry_dash_env:ImpossibleGameEnv',
    max_episode_steps=10000,
)

__all__ = ['ImpossibleGameEnv', 'FrameSkipWrapper', 'NormalizeObservation']