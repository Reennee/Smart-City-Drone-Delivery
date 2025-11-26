import gymnasium as gym
import environment
from stable_baselines3 import DQN, PPO, A2C
import torch
import argparse
import os
import sys
from environment.custom_env import CityDroneEnv
from training.reinforce_training import PolicyNetwork

def load_reinforce_model(path, env):
    model = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(path))
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo", "a2c", "reinforce"])
    parser.add_argument("--run", type=str, default="dqn_default")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    
    env = gym.make("CityDrone-v0", render_mode="human")
    
    model_path = f"models/{args.algo}/{args.run}/final_model"
    if args.algo == "reinforce":
        model_path += ".pth"
    else:
        model_path += ".zip"
        
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        # Try to find any model in that algo dir
        algo_dir = f"models/{args.algo}"
        if os.path.exists(algo_dir):
            runs = os.listdir(algo_dir)
            if runs:
                print(f"Available runs for {args.algo}: {runs}")
        return

    print(f"Loading {args.algo} model from {model_path}")
    
    if args.algo == "dqn":
        model = DQN.load(model_path)
    elif args.algo == "ppo":
        model = PPO.load(model_path)
    elif args.algo == "a2c":
        model = A2C.load(model_path)
    elif args.algo == "reinforce":
        model = load_reinforce_model(model_path, env)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        print(f"Episode {ep+1} started")
        
        while not done and not truncated:
            if args.algo == "reinforce":
                action, _ = model.select_action(obs)
            else:
                action, _ = model.predict(obs, deterministic=True)
                
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
        print(f"Episode {ep+1} finished. Reward: {total_reward:.2f}")
        
    env.close()

if __name__ == "__main__":
    main()
