import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def load_monitor_data(log_dir):
    # SB3 Monitor logs have 2 header lines usually, but we can check
    try:
        # Check if it's our custom REINFORCE log or SB3
        with open(f"{log_dir}/monitor.csv", 'r') as f:
            first_line = f.readline()
        
        if first_line.startswith("#"):
            return pd.read_csv(f"{log_dir}/monitor.csv", skiprows=1)
        else:
            return pd.read_csv(f"{log_dir}/monitor.csv")
    except Exception as e:
        print(f"Error loading {log_dir}: {e}")
        return None

def smooth(scalars, weight=0.6):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor to last smoothed value
    return smoothed

def main():
    algorithms = ["dqn", "ppo", "a2c", "reinforce"]
    base_dir = "models"
    
    plt.figure(figsize=(12, 8))
    
    for algo in algorithms:
        algo_dir = f"{base_dir}/{algo}"
        if not os.path.exists(algo_dir):
            continue
            
        # Find best run for this algo
        best_run = None
        best_reward = -float('inf')
        best_df = None

        # Explicitly check for final_exp for ALL algorithms
        # Search for any directory containing "final_exp"
        candidates = [d for d in os.listdir(algo_dir) if "final_exp" in d]
        if candidates:
             # Sort to get the most recent or simple one? Just pick the first one.
             # Ideally we want "dqn_final_exp" or "final_exp"
             best_run = candidates[0]
             # If there's an exact match "final_exp", prefer that (for PPO)
             if "final_exp" in candidates and os.path.isdir(f"{algo_dir}/final_exp"):
                 best_run = "final_exp"
                 
             df = load_monitor_data(f"{algo_dir}/{best_run}")
             if df is not None:
                 best_df = df
                 print(f"For {algo}, forcing use of: {best_run}")
        
        # If not forced or not found, search normally
        if best_run is None:
            for run_name in os.listdir(algo_dir):
                run_path = os.path.join(algo_dir, run_name)
                if not os.path.isdir(run_path) or run_name == "tensorboard" or run_name.startswith("."):
                    continue
                
                df = load_monitor_data(run_path)
                if df is not None and len(df) > 0:
                    mean_reward = df['r'].mean()
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        best_run = run_name
                        best_df = df
        
        if best_df is not None:
            # Calculate cumulative timesteps for x-axis if available, else use index
            if 't' in best_df.columns: # SB3 has 't' (time elapsed) or we can use cumulative sum of 'l' (lengths)
                # Actually SB3 monitor has 'l' (episode length) and 't' (wall time). 
                # We want total steps.
                x = np.cumsum(best_df['l'])
            else:
                x = range(len(best_df))
                
            y = smooth(best_df['r'])
            plt.plot(x, y, label=f"{algo.upper()} (Best: {best_run})")
            
    plt.title("Training Performance Comparison")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward (Smoothed)")
    plt.legend()
    plt.grid(True)
    
    output_path = "analysis/comparison_plot.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
