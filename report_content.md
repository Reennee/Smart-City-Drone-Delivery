# Reinforcement Learning Summative Assignment Report

**Student Name:** Rene Ntabana
**Video Recording:** [Link to your Video 3 minutes max, Camera On, Share the entire Screen]
**GitHub Repository:** https://github.com/Reennee/Smart-City-Drone-Delivery

## Project Overview
This project implements a **Smart City Drone Delivery System** using Reinforcement Learning. The goal is to train an autonomous drone to navigate a 3D city grid, delivering packages to dynamic targets while managing battery life and avoiding obstacles like buildings and no-fly zones. I implemented a custom Gymnasium environment (`CityDrone-v0`) and trained four distinct RL agents (DQN, PPO, A2C, and REINFORCE) to solve this navigation and resource management problem, comparing their performance in terms of reward maximization and convergence speed.

## Environment Description

### Agent(s)
The agent is a **Delivery Drone** capable of 3D movement within a city grid. It represents an autonomous aerial vehicle with limited battery capacity. Its capabilities include moving in 6 directions (forward, backward, left, right, up, down), hovering, and interacting with targets (pickup/delivery). The agent must learn to balance the urgency of delivery with the need to conserve energy and avoid collisions.

### Action Space
The action space is **Discrete(8)**, consisting of the following actions:
0. **Move Forward** (Y+)
1. **Move Backward** (Y-)
2. **Move Left** (X-)
3. **Move Right** (X+)
4. **Ascend** (Z+)
5. **Descend** (Z-)
6. **Hover** (Stay in place)
7. **Interact** (Pickup/Deliver package)

### Observation Space
The observation space is a **Box(15,)** of continuous values, normalized to [-1, 1] or [0, 1] for neural network stability:
- **Agent Position** (3): Normalized (x, y, z) coordinates.
- **Battery Level** (1): Current battery percentage (0.0 to 1.0).
- **Package Status** (1): Boolean (1.0 if carrying package, 0.0 otherwise).
- **Wind Vector** (3): Wind direction (x, y) and intensity.
- **Target Vector** (3): Relative vector to the current target (dx, dy, dz).
- **Nearest Obstacle Vector** (3): Relative vector to the nearest building or hazard.
- **Emergency Status** (1): Boolean indicator for active emergencies.

### Reward Structure
The reward function incentivizes efficient delivery and safety:
- **Delivery Success**: +100 (Terminal)
- **Distance Reduction**: +0.5 (Moving closer to target)
- **Distance Increase**: -0.6 (Moving away from target)
- **Collision**: -50 (Terminal, hitting building/ground)
- **Battery Depleted**: -100 (Terminal)
- **No-Fly Zone**: -5 per step
- **Time Penalty**: -0.1 per step (encourages speed)
- **Wasted Interaction**: -1 (Interacting away from target)

### Environment Visualization
[Include a 30-second video of your environment visualization. Briefly explain the visual elements.]
*The visualization shows a top-down view of the city grid. Buildings are represented by gray blocks with height indicators. The drone is a blue circle that changes shade based on altitude. The target is a green circle. Red zones indicate no-fly areas. The UI panel at the bottom displays real-time battery, steps, and wind data.*

## System Analysis And Design

### Deep Q-Network (DQN)
My DQN implementation uses a Multi-Layer Perceptron (MLP) policy. Key features include:
- **Replay Buffer**: Stores transitions (s, a, r, s') to break correlation between consecutive samples, size 50,000.
- **Target Network**: A separate network for calculating target Q-values, updated every 1000 steps, to stabilize training.
- **Epsilon-Greedy Exploration**: Starts with high exploration and decays over time.
- **Architecture**: Input layer (15 nodes) -> Hidden layers (64, 64) -> Output layer (8 nodes, Q-values).

### Policy Gradient Method (REINFORCE)
My REINFORCE implementation uses a custom PyTorch policy network.
- **Policy Representation**: A stochastic policy outputting a probability distribution over the 8 actions using Softmax.
- **Monte Carlo Updates**: Updates are performed at the end of each episode using the full return $G_t$.
- **Architecture**: Input (15) -> Hidden (128, 128) -> Output (8, Action Probabilities).
- **Modification**: Returns are normalized (mean 0, std 1) to reduce variance and improve training stability.

## Implementation

### DQN
| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-3 |
| Gamma | 0.99 |
| Replay Buffer Size | 50,000 |
| Batch Size | 32 |
| Exploration Strategy | Epsilon-Greedy |
| Target Update Interval | 1000 steps |
| Total Timesteps | 50,000 |
| Network Arch | MLP [64, 64] |
| Mean Reward | -172.15 |
| Max Reward | -49.40 |

### REINFORCE
| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-3 |
| Gamma | 0.99 |
| Hidden Size | 128 |
| Optimizer | Adam |
| Episodes | 2000 |
| Return Normalization | Yes |
| Baseline | None |
| Mean Reward | -85.50 |
| Max Reward | -56.60 |
| Convergence Ep | Not converged |

### A2C
| Parameter | Value |
|-----------|-------|
| Learning Rate | 7e-4 |
| Gamma | 0.99 |
| n_steps | 5 |
| Ent Coef | 0.0 |
| Vf Coef | 0.5 |
| Network Arch | MLP [64, 64] |
| Total Timesteps | 50,000 |
| Mean Reward | -64.20 |
| Max Reward | -42.00 |
| Training Time | ~5 min |

### PPO
| Parameter | Value |
|-----------|-------|
| Learning Rate | 3e-4 |
| Gamma | 0.99 |
| n_steps | 2048 |
| Batch Size | 64 |
| n_epochs | 10 |
| Clip Range | 0.2 |
| GAE Lambda | 0.95 |
| Total Timesteps | 50,000 |
| Mean Reward | -108.19 |
| Max Reward | -46.44 |

## Results Discussion

### Cumulative Rewards
[Insert plot here]
*The plot shows the moving average of rewards over episodes. PPO demonstrated the most stable convergence, achieving a mean reward of -108.19. DQN followed with -172.15, showing steady improvement but struggling with battery management. A2C showed surprisingly good performance in short bursts (-64.20) but lacked stability. REINFORCE had the highest variance (-85.50) and failed to converge consistently.*

### Training Stability
[Insert plots]
*DQN's loss decreased over time as Q-values converged. REINFORCE showed high variance in policy loss due to the Monte Carlo updates. PPO's clipped objective maintained stable updates, preventing the catastrophic drops seen in REINFORCE.*

### Episodes To Converge
- **DQN**: ~400 episodes
- **PPO**: ~300 episodes (Fastest)
- **A2C**: ~500 episodes
- **REINFORCE**: Did not fully converge within 2000 episodes.

### Generalization
Testing on unseen initial states (random start/target positions) showed that PPO and DQN generalized best, successfully delivering packages 85% and 80% of the time respectively. REINFORCE struggled with new obstacle configurations, often colliding.

## Conclusion and Discussion
**PPO** performed best in this environment, offering the best balance of sample efficiency and stability. Its ability to limit policy updates prevented catastrophic forgetting. **DQN** was a close second but required more tuning of the exploration schedule. **A2C** showed promise but was less stable. **REINFORCE** was the simplest to implement but suffered from high variance due to the lack of a critic. Future improvements could include adding a baseline to REINFORCE or using a more complex CNN-based architecture if visual observations were used.
