import gymnasium as gym
import torch
import argparse
import os
import sys
import time
import numpy as np

# Import Stable Baselines3 components
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import your custom environment and models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import environment
from training.reinforce_training import PolicyNetwork

def load_reinforce_model(path, env):
    """Loads the custom PyTorch REINFORCE model."""
    model = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    try:
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading REINFORCE model: {e}")
        sys.exit(1)

def run_sb3_eval(args, model_path, base_path):
    """
    Handles evaluation for Stable Baselines3 models (PPO, DQN, A2C).
    Uses DummyVecEnv and loads Normalization stats if available.
    """
    # 1. Create Environment wrapped in DummyVecEnv (Required for SB3)
    # We use a lambda to delay creation until inside the VecEnv
    env = DummyVecEnv([lambda: gym.make("CityDrone-v0", render_mode="human")])
    
    # Set FPS in the underlying environment
    env.envs[0].metadata["render_fps"] = args.fps

    # 2. Check for and Load Normalization Statistics (The Fix for "Stays Put" bug)
    stats_path = os.path.join(base_path, "vec_normalize.pkl")
    if os.path.exists(stats_path):
        print(f"[-] Loading normalization stats from: {stats_path}")
        env = VecNormalize.load(stats_path, env)
        env.training = False   # Do not update stats during test
        env.norm_reward = False # Do not normalize rewards during test
    else:
        print("[!] Warning: vec_normalize.pkl not found. Agent may behave erratically.")

    # 3. Load Model
    print(f"[-] Loading {args.algo.upper()} model from: {model_path}")
    if args.algo == "dqn":
        model = DQN.load(model_path, env=env)
    elif args.algo == "ppo":
        model = PPO.load(model_path, env=env)
    elif args.algo == "a2c":
        model = A2C.load(model_path, env=env)

    # 4. Evaluation Loop
    for ep in range(args.episodes):
        obs = env.reset() # VecEnv reset returns only obs
        done = False
        total_reward = 0
        step_count = 0
        
        print(f"--- Episode {ep+1} ---")
        
        while not done:
            # SB3 predict returns (action, state)
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            
            # VecEnv step returns (obs, reward, done, info)
            # Note: 'done' here is an array of booleans, usually [True] or [False]
            obs, reward, dones, infos = env.step(action)
            
            total_reward += reward[0]
            step_count += 1
            env.render() # Explicit render call for VecEnv
            
            # Check if the episode is actually done
            if dones[0]:
                done = True

        print(f"    Finished. Steps: {step_count} | Total Reward: {total_reward:.2f}")
        time.sleep(1.0)
    
    env.close()

def run_custom_eval(args, model_path):
    """
    Handles evaluation for custom PyTorch models (REINFORCE).
    Uses standard Gym API (no VecEnv).
    """
    env = gym.make("CityDrone-v0", render_mode="human")
    env.metadata["render_fps"] = args.fps
    
    print(f"[-] Loading REINFORCE model from: {model_path}")
    model = load_reinforce_model(model_path, env)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        print(f"--- Episode {ep+1} ---")
        
        while not done and not truncated:
            # Custom select_action
            action, _ = model.select_action(obs)
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            # Render is handled automatically by 'human' mode in standard gym, 
            # but we can force it if needed:
            # env.render()
            
        print(f"    Finished. Steps: {step_count} | Total Reward: {total_reward:.2f}")
        time.sleep(1.0)

    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo", "a2c", "reinforce"])
    parser.add_argument("--run", type=str, default="run_1", help="Name of the run folder")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true", help="Use random actions (good for debugging stuck agents)")
    args = parser.parse_args()
    
    # Construct paths
    base_path = os.path.join("models", args.algo, args.run)
    filename = "final_model"
    
    if args.algo == "reinforce":
        model_path = os.path.join(base_path, filename + ".pth")
    else:
        model_path = os.path.join(base_path, filename + ".zip")
        
    if not os.path.exists(model_path):
        print(f"[!] Model not found: {model_path}")
        return

    # Branch based on Algorithm Type
    if args.algo in ["dqn", "ppo", "a2c"]:
        run_sb3_eval(args, model_path, base_path)
    else:
        run_custom_eval(args, model_path)

if __name__ == "__main__":
    main()