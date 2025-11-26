import gymnasium as gym
import environment
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import argparse
import time

def train_dqn(learning_rate=1e-3, buffer_size=50000, batch_size=32, gamma=0.99, 
              target_update_interval=1000, total_timesteps=50000, run_name="dqn_run"):
    
    # Create logs directory
    log_dir = f"models/dqn/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    env = gym.make("CityDrone-v0")
    env = Monitor(env, log_dir)
    
    # Create model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        target_update_interval=target_update_interval,
        verbose=1,
        tensorboard_log="models/dqn/tensorboard"
    )
    
    # Callbacks
    eval_env = gym.make("CityDrone-v0")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    # Train
    print(f"Starting training: {run_name}")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, tb_log_name=run_name)
    end_time = time.time()
    
    # Save final model
    model.save(f"{log_dir}/final_model")
    print(f"Training finished in {end_time - start_time:.2f}s")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target_update", type=int, default=1000)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--name", type=str, default="dqn_default")
    
    args = parser.parse_args()
    
    train_dqn(
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_update_interval=args.target_update,
        total_timesteps=args.timesteps,
        run_name=args.name
    )
