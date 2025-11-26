import gymnasium as gym
import environment
import numpy as np
import pygame
import os

def main():
    # Create environment
    env = gym.make("CityDrone-v0", render_mode="human")
    
    # Reset
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step = 0
    
    print("Starting Random Agent Demo...")
    print(f"Initial State: {obs}")
    print(f"Target: {info['distance']:.2f} units away")
    
    # Run episode
    while not done and not truncated:
        # Random action
        action = env.action_space.sample()
        
        # Step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        if step % 10 == 0:
            print(f"Step {step}: Action={action}, Reward={reward:.2f}, Battery={info['battery']:.1f}")
            
        # Slow down for visualization
        pygame.time.wait(50)
        
        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Episode Finished. Total Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()
