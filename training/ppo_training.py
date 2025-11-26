import gymnasium as gym
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import environment
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import argparse
import time

def train_ppo(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, 
              gamma=0.99, gae_lambda=0.95, clip_range=0.2, 
              total_timesteps=50000, run_name="ppo_run"):
    
    log_dir = f"models/ppo/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    env = gym.make("CityDrone-v0")
    env = Monitor(env, log_dir)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=1,
        tensorboard_log="models/ppo/tensorboard"
    )
    
    eval_env = gym.make("CityDrone-v0")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    print(f"Starting training: {run_name}")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, tb_log_name=run_name)
    end_time = time.time()
    
    model.save(f"{log_dir}/final_model")
    print(f"Training finished in {end_time - start_time:.2f}s")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--name", type=str, default="ppo_default")
    
    args = parser.parse_args()
    
    train_ppo(
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        total_timesteps=args.timesteps,
        run_name=args.name
    )
