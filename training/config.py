"""
Training configurations for all RL agents.
"""

# Q-Learning config (off-policy value-based)
Q_LEARNING_CONFIG = {
    'learning_rate': 0.1,
    'discount_factor': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.9995,
    'epsilon_min': 0.01,
    'bins': 10,
    'episodes': 5000,
    'max_steps': 10000
}

# SARSA config (on-policy value-based)
SARSA_CONFIG = {
    'learning_rate': 0.1,
    'discount_factor': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.9995,
    'epsilon_min': 0.01,
    'bins': 10,
    'episodes': 5000,
    'max_steps': 10000
}

# DQN config (deep value-based with experience replay)
# --- CONFIGURARE PENTRU PERFORMANȚĂ ÎNALTĂ ---
DQN_CONFIG = {
    'episodes': 30000,          # Mărim numărul de episoade (antrenamentul cere timp)
    'max_steps': 2500,          
    'learning_rate': 0.0001,    # MAI MIC! (Esențial pentru precizie "pixel-perfect")
    'discount_factor': 0.995,   # Viziune pe termen mai lung (nu sari dacă aterizezi în țepi)
    'epsilon': 1.0,
    'epsilon_decay': 0.999999,    # Scădere FOARTE lentă. Trebuie să exploreze mii de episoade.
    'epsilon_min': 0.01,        # Lasă-l să mai încerce chestii random uneori
    'buffer_size': 200000,      # Memorie mai mare (dacă ai RAM, altfel 100k)
    'batch_size': 256,          # Batch mare = învățare mai stabilă
    'target_update_freq': 4000  # Actualizăm rețeaua țintă mai rar (stabilitate)
}

# PPO config (policy gradient with clipped objective)
# MODIFICAT: ent_coef crescut pentru explorare și steps ajustați
PPO_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.02,           # Crescut de la 0.01 la 0.02 pentru a evita stagnarea
    'total_timesteps': 1000000  # PPO are nevoie de mult timp (aprox 1h-2h pe CPU decent)
}

# Hyperparameter experiment configs
EXPERIMENT_CONFIGS = {
    'learning_rates': [0.01, 0.05, 0.1, 0.2],
    'discount_factors': [0.9, 0.95, 0.99],
    'epsilon_decays': [0.999, 0.9995, 0.9999],
    'dqn_batch_sizes': [32, 64, 128],
    'dqn_learning_rates': [1e-5, 1e-4, 1e-3],
}