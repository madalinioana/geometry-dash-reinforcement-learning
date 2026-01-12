# Reinforcement Learning - Geometry Dash Agent

Un proiect de Reinforcement Learning care antreneaza agenti inteligenti pentru a juca un joc similar cu Geometry Dash/The Impossible Game.

## Cuprins

- [Descriere](#descriere)
- [Cerinte Proiect](#cerinte-proiect)
- [Structura Proiectului](#structura-proiectului)
- [Environment](#environment)
- [Algoritmi Implementati](#algoritmi-implementati)
- [Instalare](#instalare)
- [Utilizare](#utilizare)
- [Rezultate](#rezultate)
- [Experimente Hyperparameter Tuning](#experimente-hyperparameter-tuning)
- [Referinte](#referinte)

---

## Descriere

Acest proiect implementeaza si compara 4 algoritmi de Reinforcement Learning pentru a antrena agenti care invata sa joace un joc de tip platformer 2D. Jocul este inspirat din Geometry Dash/The Impossible Game, unde jucatorul trebuie sa evite obstacole prin sarituri bine calculate.

**Obiective:**
- Implementarea unui environment custom compatibil cu Gymnasium
- Implementarea si compararea a 4 algoritmi RL (2 tabulari + 2 deep learning)
- Analiza hiperparametrilor si impactul lor asupra performantei
- Vizualizarea si interpretarea rezultatelor

---

## Cerinte Proiect

| Cerinta | Status |
|---------|--------|
| Environment custom/modificat | Implementat |
| Minim 3 algoritmi | 4 algoritmi (Q-Learning, SARSA, DQN, PPO) |
| Comparatie in acelasi mediu | Implementat |
| Reglare hiperparametri | Experimente multiple |
| Grafice si metrici | Vizualizari comprehensive |
| Documentatie completa | README + comentarii cod |

---

## Structura Proiectului

```
Rl-GeometryDash/
├── main.py                    # Script principal de orchestrare
├── requirements.txt           # Dependinte Python
├── README.md                  # Documentatia proiectului
│
├── environment/               # Gymnasium Environment
│   ├── __init__.py
│   ├── geometry_dash_env.py   # Implementarea jocului
│   └── wrappers.py            # Environment wrappers
│
├── agents/                    # Implementarile agentilor RL
│   ├── base_agent.py          # Clasa de baza abstracta
│   ├── tabular/               # Metode tabulare
│   │   ├── q_learning_agent.py
│   │   └── sarsa_agent.py
│   ├── deep/                  # Deep RL
│   │   ├── dqn_agent.py
│   │   └── replay_buffer.py
│   └── policy/                # Policy Gradient
│       └── ppo_agent.py
│
├── training/                  # Scripturi de antrenare
│   ├── config.py              # Configurari hiperparametri
│   ├── train_q_learning.py
│   ├── train_sarsa.py
│   ├── train_dqn.py
│   └── train_ppo.py
│
├── evaluation/                # Evaluare si comparatie
│   ├── evaluate.py
│   ├── compare_agents.py
│   └── visualize_agents.py
│
├── analysis/                  # Analiza rezultatelor
│   └── plot_results.py
│
├── experiments/               # Experimente stiintifice
│   └── hyperparameter_tuning.py
│
└── results/                   # Output (generat automat)
    ├── models/                # Modele antrenate
    ├── logs/                  # Metrici de training
    ├── plots/                 # Grafice
    └── experiments/           # Rezultate experimente
```

---

## Environment

### Geometry Dash Environment

Un joc de tip platformer 2D implementat ca environment Gymnasium.

**Mecanica jocului:**
- Jucatorul (patrat albastru) se deplaseaza automat spre dreapta
- Obstacole generate procedural: spike-uri (rosu), goluri, platforme (galben)
- Dificultatea creste progresiv (viteza)
- Obiectiv: supravietuire cat mai mult timp

**Spatiul de observatie (23 dimensiuni):**
```
[player_y, player_vel_y, on_ground,
 obs1_type, obs1_x, obs1_y, obs1_width,
 obs2_type, obs2_x, obs2_y, obs2_width,
 ... (5 obstacole)]
```

**Spatiul de actiuni:**
- `0`: Nu face nimic
- `1`: Saritura

**Sistemul de recompense:**
- `+1.0` per frame supravietuit
- `+10.0` bonus pentru depasirea unui obstacol
- `-100.0` penalizare la moarte

### Wrappers disponibile:
- `FrameSkipWrapper`: Agent decide la fiecare N frame-uri
- `NormalizeObservation`: Normalizeaza observatiile
- `RewardShapingWrapper`: Recompense aditionale pentru distanta

---

## Algoritmi Implementati

### 1. Q-Learning (Tabular, Off-Policy)

**Algoritm clasic value-based care invata Q-values pentru perechi (state, action).**

```
Q(s,a) <- Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

**Caracteristici:**
- Off-policy: foloseste max Q pentru update
- Discretizeaza spatiul de stari (10 bins)
- Epsilon-greedy exploration
- Q-table sparse (dictionar)

**Hiperparametri default:**
- Learning rate: 0.1
- Discount factor: 0.99
- Epsilon decay: 0.9995
- Episoade: 5000

### 2. SARSA (Tabular, On-Policy)

**Similar cu Q-Learning, dar on-policy - foloseste actiunea efectiv executata.**

```
Q(s,a) <- Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
```

**Diferenta cheie fata de Q-Learning:**
- Mai conservator
- Tine cont de politica de explorare
- Mai sigur in medii cu penalizari mari

### 3. DQN - Deep Q-Network (Deep, Off-Policy)

**Aproximeaza functia Q cu o retea neurala.**

**Arhitectura:**
```
Input(23) -> Linear(128) -> ReLU
          -> Linear(128) -> ReLU
          -> Linear(64)  -> ReLU
          -> Linear(2)   -> Q-values
```

**Tehnici cheie:**
- **Experience Replay**: Buffer de 100k tranzitii, batch sampling
- **Target Network**: Retea separata pentru stabilitate (update la 1000 pasi)
- **Epsilon-greedy**: Explorare cu decay

**Hiperparametri:**
- Learning rate: 1e-4 (Adam optimizer)
- Batch size: 64
- Target update: 1000 steps
- Episoade: 2000

### 4. PPO - Proximal Policy Optimization (Deep, On-Policy)

**Algoritm policy gradient cu obiectiv clipat pentru stabilitate.**

**Implementare:** Stable-Baselines3 wrapper

**Caracteristici:**
- Actor-Critic architecture
- Clipped surrogate objective
- GAE (Generalized Advantage Estimation)
- Multiple epochs per batch

**Hiperparametri:**
- Learning rate: 3e-4
- N-steps: 2048
- Batch size: 64
- Epochs: 10
- Clip range: 0.2
- GAE lambda: 0.95
- Entropy coefficient: 0.01

---

## Instalare

### Cerinte sistem:
- Python 3.9+
- CUDA (optional, pentru accelerare GPU)

### Pasi instalare:

```bash
# 1. Cloneaza repository-ul
git clone <repository_url>
cd Rl-GeometryDash

# 2. Creeaza environment virtual (recomandat)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# sau
venv\Scripts\activate     # Windows

# 3. Instaleaza dependintele
pip install -r requirements.txt
```

### Dependinte principale:
- `gymnasium>=0.29.0`
- `stable-baselines3>=2.2.0`
- `torch>=2.0.0`
- `pygame>=2.5.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `pandas>=2.0.0`
- `tqdm>=4.65.0`

---

## Utilizare

### Script principal (main.py)

```bash
# Afiseaza help
python main.py --help

# Antreneaza toti agentii
python main.py --train all

# Antreneaza un agent specific
python main.py --train q_learning
python main.py --train sarsa
python main.py --train dqn
python main.py --train ppo

# Evalueaza agentii antrenati
python main.py --evaluate

# Compara agentii si genereaza grafice
python main.py --compare

# Genereaza grafice de training
python main.py --plots

# Ruleaza experimente de hyperparameter tuning
python main.py --experiments

# Demo: urmareste un agent jucand
python main.py --demo ppo

# Pipeline complet: train + evaluate + compare
python main.py --full
```

### Antrenare individuala

```bash
# Q-Learning
python training/train_q_learning.py

# SARSA
python training/train_sarsa.py

# DQN
python training/train_dqn.py

# PPO
python training/train_ppo.py
```

### Evaluare si vizualizare

```bash
# Evaluare completa
python evaluation/evaluate.py

# Comparatie agenti
python evaluation/compare_agents.py

# Vizualizare agent jucand
python evaluation/visualize_agents.py --agent ppo --episodes 5
```

---

## Rezultate

### Metrici de performanta

Rezultatele sunt salvate in `results/`:

| Fisier | Descriere |
|--------|-----------|
| `comparison_table.csv` | Tabel comparativ agenti |
| `detailed_comparison.csv` | Statistici detaliate |
| `plots/agents_comparison.png` | Grafice comparative |
| `plots/training_curves_*.png` | Curbe de invatare |
| `plots/convergence_analysis.png` | Analiza convergentei |

### Grafice generate:

1. **Training Curves** - Evolutia reward/score pe parcursul antrenarii
2. **Score Comparison** - Comparatie medie scoruri cu error bars
3. **Score Distribution** - Box plot cu distributia scorurilor
4. **Convergence Analysis** - Analiza stabilitatii si convergentei

### Interpretare rezultate:

- **Mean Score**: Scorul mediu obtinut pe 100 episoade de evaluare
- **Std Score**: Variabilitatea performantei
- **Max Score**: Cel mai bun scor obtinut
- **Mean Reward**: Recompensa totala medie per episod
- **Mean Length**: Durata medie a episoadelor (in steps)

---

## Experimente Hyperparameter Tuning

Scriptul `experiments/hyperparameter_tuning.py` ruleaza urmatoarele experimente:

### Experiment 1: Learning Rate
Testeaza impactul ratei de invatare: `[0.01, 0.05, 0.1, 0.2, 0.5]`

### Experiment 2: Discount Factor
Testeaza impactul factorului gamma: `[0.8, 0.9, 0.95, 0.99, 0.999]`

### Experiment 3: Exploration Decay
Testeaza viteza de decay a explorarii: `[0.99, 0.995, 0.999, 0.9995, 0.9999]`

### Experiment 4: Q-Learning vs SARSA
Comparatie directa cu aceiasi hiperparametri.

**Output:**
- `results/experiments/hyperparameter_summary.csv` - Rezumat
- `results/experiments/*.png` - Grafice pentru fiecare experiment

---

## Probleme intampinate si solutii

| Problema | Solutie |
|----------|---------|
| Discretizarea starii pentru metode tabulare | Folosirea doar a caracteristicilor esentiale (y, vel, ground) |
| Instabilitate DQN | Target network + experience replay |
| Explorare insuficienta | Epsilon decay gradual, entropy bonus (PPO) |
| Reward sparse | Bonus pentru depasirea obstacolelor |
| Dificultate variabila | Crestere progresiva a vitezei jocului |

---

## Concluzii

### Comparatie algoritmi:

| Aspect | Q-Learning | SARSA | DQN | PPO |
|--------|------------|-------|-----|-----|
| Complexitate | Scazuta | Scazuta | Medie | Ridicata |
| Sample efficiency | Medie | Medie | Ridicata | Scazuta |
| Stabilitate | Medie | Ridicata | Medie | Ridicata |
| Performanta finala | Medie | Medie | Ridicata | Foarte ridicata |

### Observatii:
- **PPO** obtine de obicei cele mai bune rezultate datorita arhitecturii Actor-Critic
- **DQN** beneficiaza de reprezentarea neuronala pentru stari continue
- **SARSA** este mai conservator decat Q-Learning (avantaj in medii cu penalizari mari)
- **Q-Learning** este rapid de antrenat dar limitat de discretizare

---

## Referinte

1. Sutton & Barto - *Reinforcement Learning: An Introduction* (2018)
   - https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf

2. Mnih et al. - *Human-level control through deep reinforcement learning* (2015)
   - https://www.nature.com/articles/nature14236

3. Schulman et al. - *Proximal Policy Optimization Algorithms* (2017)
   - https://arxiv.org/abs/1707.06347

4. Stable-Baselines3 Documentation
   - https://stable-baselines3.readthedocs.io/

5. Gymnasium Documentation
   - https://gymnasium.farama.org/

---

## Autori
Andrei Madalin Ioana
Stancu Rares
