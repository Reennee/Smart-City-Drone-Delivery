import gymnasium as gym
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import environment
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import argparse
import time

def train_a2c(learning_rate=7e-4, n_steps=5, gamma=0.99, ent_coef=0.0, 
              vf_coef=0.5, total_timesteps=50000, run_name="a2c_run"):
    
    log_dir = f"models/a2c/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    env = gym.make("CityDrone-v0")
    env = Monitor(env, log_dir)
    
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=1,
        tensorboard_log="models/a2c/tensorboard"
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
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--name", type=str, default="a2c_default")
    
    args = parser.parse_args()
    
    train_a2c(
        learning_rate=args.lr,
        n_steps=args.n_steps,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        total_timesteps=args.timesteps,
        run_name=args.name
    )
