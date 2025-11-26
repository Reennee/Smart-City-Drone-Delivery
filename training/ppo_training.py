import gymnasium as gym
import sys
import os
import time
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import environment  # Registers CityDrone-v0
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

def train_ppo(learning_rate=3e-4, total_timesteps=200000, run_name="ppo_run"):
    
    log_dir = f"models/ppo/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # --- CRITICAL FIX 1: Use DummyVecEnv & VecNormalize ---
    # This scales all inputs to a standard range, helping the agent learn 10x faster.
    env = DummyVecEnv([lambda: gym.make("CityDrone-v0")])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="models/ppo/tensorboard"
    )
    
    # Separate evaluation environment (also needs wrapping)
    eval_env = DummyVecEnv([lambda: gym.make("CityDrone-v0")])
    # We must use the SAME normalization stats as the training env
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    print(f"Starting training: {run_name} for {total_timesteps} steps...")
    start_time = time.time()
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, tb_log_name=run_name)
    
    # --- CRITICAL FIX 2: Save the Normalization Stats ---
    # Without this file, the agent acts blind during inference
    model.save(f"{log_dir}/final_model")
    env.save(f"{log_dir}/vec_normalize.pkl") 
    
    print(f"Training finished in {time.time() - start_time:.2f}s")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200000) # Increased default
    parser.add_argument("--name", type=str, default="ppo_fixed")
    args = parser.parse_args()
    
    train_ppo(total_timesteps=args.timesteps, run_name=args.name)