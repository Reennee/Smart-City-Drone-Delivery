# Smart City Drone Delivery RL Project

## Overview
This project implements a Reinforcement Learning agent for a Smart City Drone Delivery system. The agent navigates a 3D city grid to deliver packages while managing battery life, avoiding obstacles (buildings, no-fly zones), and handling dynamic wind conditions.

## Environment: CityDrone-v0
A custom Gymnasium environment simulating a drone in a city.

### Observation Space
- Drone Position (x, y, z)
- Battery Level
- Package Status
- Wind Conditions
- Nearest Obstacle Vector
- Target Position Vector
- Emergency Status

### Action Space
8 Discrete Actions:
- Move Forward/Backward/Left/Right
- Ascend/Descend
- Hover
- Interact (Pickup/Dropoff/Charge)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Random Agent Demo
```bash
python demo_random_agent.py
```

### Training
```bash
python training/dqn_training.py
python training/ppo_training.py
# ... etc
```

### Run Best Model
```bash
python main.py --algo ppo --run final_exp --episodes 10
```
