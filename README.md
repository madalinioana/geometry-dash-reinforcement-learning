# Geometry Dash Reinforcement Learning Agents

This project implements and compares multiple reinforcement learning algorithms to master a Geometry Dash style platformer game.

## Overview

The goal is to train autonomous agents to navigate through obstacle courses by learning optimal jumping strategies. The environment features dynamic obstacles, scoring systems and increasing difficulty that challenges different RL approaches.

## Results

The DQN agent achieved the best overall performance after 30,000 training episodes, successfully navigating complex obstacle patterns with consistent high scores.

## Approach

I experimented with four different reinforcement learning algorithms:

**Q-Learning**: A tabular, off-policy method serving as the baseline. Uses discrete state-action value functions for decision making.

**SARSA**: A tabular, on-policy alternative to Q-Learning. Updates policies based on actual actions taken rather than optimal actions.

**Deep Q-Network (DQN)**: The final model using deep neural networks with experience replay and target networks. Handles the high-dimensional state space more effectively than tabular methods.

**Proximal Policy Optimization (PPO)**: A policy gradient method with clipped objective for stable training. Directly optimizes the policy without value function approximation.

The DQN agent significantly outperformed the tabular methods and demonstrated the most consistent learning progress across training episodes.

## Project Structure

```
├── main.py                     # Entry point for training, evaluation and demos
├── requirements.txt            # Python dependencies
├── agents/
│   ├── base_agent.py           # Base agent interface
│   ├── tabular/                # Q-Learning and SARSA implementations
│   ├── deep/                   # DQN with experience replay
│   └── policy/                 # PPO implementation
├── environment/
│   ├── geometry_dash_env.py    # Custom Gymnasium environment
│   └── wrappers.py             # Environment wrappers
├── training/                   # Training scripts for each agent
├── evaluation/                 # Agent comparison and evaluation
├── analysis/
│   ├── plots.py                # Plotting utilities
│   ├── visualizations.py       # Visualization generation
│   └── statistics.py           # Statistics and metrics
└── results/
    ├── models/                 # Saved model checkpoints
    ├── logs/                   # Training metrics
    └── plots/                  # Generated visualizations
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train a specific agent
python main.py --train dqn

# Watch a trained agent play
python main.py --demo dqn

# Generate performance plots
python main.py --plots

# Play manually (SPACE to jump)
python main.py --play
```
