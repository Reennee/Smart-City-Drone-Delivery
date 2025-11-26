import os
import subprocess
import itertools
import sys
import random

def run_experiment(script_path, params, run_name):
    cmd = ["python3.10", script_path, "--name", run_name]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    algorithms = {
        "dqn": "training/dqn_training.py",
        "ppo": "training/ppo_training.py",
        "a2c": "training/a2c_training.py",
        "reinforce": "training/reinforce_training.py"
    }
    
    param_grids = {
        "dqn": {
            "lr": [1e-3, 5e-4, 1e-4],
            "batch_size": [32, 64],
            "timesteps": [5000]
        },
        "ppo": {
            "lr": [3e-4, 1e-4, 5e-5],
            "n_steps": [1024, 2048],
            "timesteps": [5000]
        },
        "a2c": {
            "lr": [7e-4, 1e-4, 5e-4],
            "n_steps": [5, 10],
            "timesteps": [5000]
        },
        "reinforce": {
            "lr": [1e-3, 5e-4, 1e-4],
            "episodes": [200]
        }
    }

    # Generate 10 experiments per algorithm
    for algo, script in algorithms.items():
        print(f"Starting 10 {algo.upper()} experiments...")
        
        for i in range(10):
            params = {}
            for param, values in param_grids[algo].items():
                params[param] = random.choice(values)
            
            run_experiment(script, params, f"{algo}_exp_{i}")

if __name__ == "__main__":
    main()
