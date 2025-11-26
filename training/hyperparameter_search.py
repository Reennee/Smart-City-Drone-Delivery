import os
import subprocess
import itertools

def run_experiment(script, params, run_name):
    cmd = ["python", script, "--name", run_name]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    # DQN Experiments
    dqn_params = {
        "lr": [1e-3, 1e-4],
        "batch_size": [32, 64],
        "gamma": [0.99],
        "timesteps": [20000] # Reduced for quick demo, increase for real
    }
    
    # Generate combinations
    keys, values = zip(*dqn_params.items())
    dqn_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Starting {len(dqn_combinations)} DQN experiments...")
    for i, params in enumerate(dqn_combinations):
        run_experiment("training/dqn_training.py", params, f"dqn_exp_{i}")

    # PPO Experiments
    ppo_params = {
        "lr": [3e-4, 1e-4],
        "n_steps": [1024, 2048],
        "timesteps": [20000]
    }
    keys, values = zip(*ppo_params.items())
    ppo_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Starting {len(ppo_combinations)} PPO experiments...")
    for i, params in enumerate(ppo_combinations):
        run_experiment("training/ppo_training.py", params, f"ppo_exp_{i}")

    # A2C Experiments
    a2c_params = {
        "lr": [7e-4, 1e-4],
        "n_steps": [5, 10],
        "timesteps": [20000]
    }
    keys, values = zip(*a2c_params.items())
    a2c_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Starting {len(a2c_combinations)} A2C experiments...")
    for i, params in enumerate(a2c_combinations):
        run_experiment("training/a2c_training.py", params, f"a2c_exp_{i}")

    # REINFORCE Experiments
    reinforce_params = {
        "lr": [1e-3, 5e-4],
        "episodes": [500] # Short for demo
    }
    keys, values = zip(*reinforce_params.items())
    reinforce_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Starting {len(reinforce_combinations)} REINFORCE experiments...")
    for i, params in enumerate(reinforce_combinations):
        run_experiment("training/reinforce_training.py", params, f"reinforce_exp_{i}")

if __name__ == "__main__":
    main()
