Q_LEARNING_CONFIG = {
    'learning_rate': 0.2,
    'discount_factor': 0.98,
    'epsilon': 1.0,
    'epsilon_decay': 0.9995,
    'epsilon_min': 0.01,
    'bins': 12,
    'episodes': 15000,
    'max_steps': 10000
}

SARSA_CONFIG = {
    'learning_rate': 0.2,
    'discount_factor': 0.98,
    'epsilon': 1.0,
    'epsilon_decay': 0.9995,
    'epsilon_min': 0.01,
    'bins': 12,
    'episodes': 15000,
    'max_steps': 10000
}

DQN_CONFIG = {
    'episodes': 30000,
    'max_steps': 2500,          
    'learning_rate': 0.0001,
    'discount_factor': 0.995,
    'epsilon': 1.0,
    'epsilon_decay': 0.999999,
    'epsilon_min': 0.01,
    'buffer_size': 200000,
    'batch_size': 256,
    'target_update_freq': 4000
}

PPO_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.02,
    'total_timesteps': 1000000
}

EXPERIMENT_CONFIGS = {
    'learning_rates': [0.01, 0.05, 0.1, 0.2],
    'discount_factors': [0.9, 0.95, 0.99],
    'epsilon_decays': [0.999, 0.9995, 0.9999],
    'dqn_batch_sizes': [32, 64, 128],
    'dqn_learning_rates': [1e-5, 1e-4, 1e-3],
}