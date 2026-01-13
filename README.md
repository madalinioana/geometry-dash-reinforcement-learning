# RL Geometry Dash

Reinforcement Learning agents trained to play a Geometry Dash style game.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Play manually

```bash
python main.py --play
```

Controls: SPACE to jump

### Train agents

```bash
# Train all agents
python main.py --train all

# Train specific agent
python main.py --train dqn
python main.py --train q_learning
python main.py --train sarsa
python main.py --train ppo
```

### Watch trained agents

```bash
python main.py --demo dqn
python main.py --demo q_learning
python main.py --demo sarsa
python main.py --demo ppo
```

### Generate plots

```bash
python main.py --plots
```

Plots saved in `results/plots/`

## Project Structure

```
agents/          # RL agent implementations
  tabular/       # Q-Learning, SARSA
  deep/          # DQN
  policy/        # PPO
environment/     # Custom Gymnasium environment
training/        # Training scripts
evaluation/      # Evaluation and comparison
analysis/        # Plot generation
results/         # Saved models, logs, plots
```

## Agents Implemented

- **Q-Learning** - Tabular, off-policy
- **SARSA** - Tabular, on-policy
- **DQN** - Deep Q-Network with experience replay
- **PPO** - Proximal Policy Optimization

## Results

Training results and model checkpoints are saved in:

- `results/models/` - Trained models
- `results/logs/` - Training logs (.npy files)
- `results/plots/` - Visualizations
- `results/*.csv` - Comparison tables
