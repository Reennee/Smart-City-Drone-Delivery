import gymnasium as gym
import environment
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import time

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

def train_reinforce(learning_rate=1e-3, gamma=0.99, hidden_size=128, 
                    total_episodes=2000, run_name="reinforce_run"):
    
    log_dir = f"models/reinforce/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f"models/reinforce/tensorboard/{run_name}")
    
    env = gym.make("CityDrone-v0")
    
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, hidden_size)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    print(f"Starting training: {run_name}")
    start_time = time.time()
    
    for episode in range(total_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        # Run episode
        while True:
            action, log_prob = policy.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            
            if done or truncated:
                break
                
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
            
        returns = torch.tensor(returns)
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        # Update policy
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
            
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        # Logging
        total_reward = sum(rewards)
        writer.add_scalar("rollout/ep_rew_mean", total_reward, episode)
        writer.add_scalar("train/loss", policy_loss.item(), episode)
        
        # CSV Logging
        with open(f"{log_dir}/monitor.csv", "a") as f:
            f.write(f"{episode},{total_reward},{len(rewards)}\n")
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward={total_reward:.2f}")
            
    end_time = time.time()
    
    # Save model
    torch.save(policy.state_dict(), f"{log_dir}/final_model.pth")
    print(f"Training finished in {end_time - start_time:.2f}s")
    writer.close()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--name", type=str, default="reinforce_default")
    
    args = parser.parse_args()
    
    # Initialize CSV
    log_dir = f"models/reinforce/{args.name}"
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/monitor.csv", "w") as f:
        f.write("episode,r,l\n") # Header to match SB3 monitor somewhat (r=reward, l=length)
    
    train_reinforce(
        learning_rate=args.lr,
        gamma=args.gamma,
        hidden_size=args.hidden_size,
        total_episodes=args.episodes,
        run_name=args.name
    )
