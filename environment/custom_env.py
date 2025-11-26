import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class CityDroneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Environment dimensions
        self.grid_size = 12
        self.max_altitude = 5
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.renderer = None

        # Action Space: 8 discrete actions
        # 0: Forward, 1: Backward, 2: Left, 3: Right
        # 4: Ascend, 5: Descend, 6: Hover, 7: Interact
        self.action_space = spaces.Discrete(8)

        # Observation Space
        # Normalized values for neural network stability
        # 0-2: Position (x, y, z)
        # 3: Battery
        # 4: Package Status
        # 5-7: Wind Vector (x, y, intensity)
        # 8-10: Target Vector (dx, dy, dz)
        # 11-13: Nearest Obstacle Vector
        # 14: Emergency Status
        # 15-17: Velocity/Previous Action (optional, keeping simple for now)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(15,), dtype=np.float32
        )

        # Environment State
        self._agent_location = np.array([0, 0, 0], dtype=int)
        self._target_location = np.array([0, 0, 0], dtype=int)
        self._battery = 100.0
        self._has_package = False
        self._wind_vector = np.array([0, 0, 0.0], dtype=float) # dx, dy, intensity
        self._emergency_active = False
        self._buildings = self._generate_buildings()
        self._no_fly_zones = self._generate_no_fly_zones()
        
        # Constants
        self.BATTERY_DRAIN_MOVE = 0.5
        self.BATTERY_DRAIN_HOVER = 0.1
        self.BATTERY_DRAIN_WIND = 0.2
        self.BATTERY_DRAIN_CARRY = 0.2
        self.MAX_STEPS = 500
        self._current_step = 0

    def _generate_buildings(self):
        # Generate random buildings with varying heights
        # Grid representation: height of building at x,y
        buildings = np.zeros((self.grid_size, self.grid_size), dtype=int)
        num_buildings = 15
        for _ in range(num_buildings):
            x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            if (x, y) != (0, 0): # Don't block start
                buildings[x, y] = random.randint(1, self.max_altitude - 1)
        return buildings

    def _generate_no_fly_zones(self):
        zones = []
        # Add 1-2 restricted areas
        for _ in range(2):
            x = random.randint(0, self.grid_size-3)
            y = random.randint(0, self.grid_size-3)
            zones.append((x, y, x+2, y+2)) # Rectangular zones
        return zones

    def _get_obs(self):
        # Normalize observations
        pos_norm = self._agent_location / [self.grid_size, self.grid_size, self.max_altitude]
        target_vec = (self._target_location - self._agent_location) / self.grid_size
        
        # Find nearest obstacle
        nearest_obs_dist = 100
        nearest_obs_vec = np.zeros(3)
        
        # Simple nearest building check
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                h = self._buildings[x, y]
                if h > 0:
                    # Check distance to building blocks
                    for z in range(h):
                        dist = np.linalg.norm(self._agent_location - np.array([x, y, z]))
                        if dist < nearest_obs_dist:
                            nearest_obs_dist = dist
                            nearest_obs_vec = (np.array([x, y, z]) - self._agent_location) / self.grid_size

        obs = np.concatenate([
            pos_norm,
            [self._battery / 100.0],
            [1.0 if self._has_package else 0.0],
            self._wind_vector,
            target_vec,
            nearest_obs_vec,
            [1.0 if self._emergency_active else 0.0]
        ])
        return obs.astype(np.float32)

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._target_location - self._agent_location),
            "battery": self._battery,
            "has_package": self._has_package
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._agent_location = np.array([0, 0, 0], dtype=int)
        self._battery = 100.0
        self._has_package = False
        self._current_step = 0
        self._emergency_active = False
        
        # Random target (not at start, accessible)
        while True:
            tx, ty = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            if (tx, ty) != (0, 0) and self._buildings[tx, ty] == 0:
                self._target_location = np.array([tx, ty, 0], dtype=int) # Ground level delivery
                break
                
        # Random wind
        self._wind_vector = np.array([
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(0, 1) # Intensity
        ])
        
        if self.render_mode == "human" and self.renderer is None:
            from environment.rendering import CityDroneRenderer
            self.renderer = CityDroneRenderer(self)

        return self._get_obs(), self._get_info()

    def step(self, action):
        self._current_step += 1
        reward = 0
        terminated = False
        truncated = False
        
        # Movement logic
        prev_location = self._agent_location.copy()
        direction = np.array([0, 0, 0])
        
        if action == 0: direction = np.array([0, 1, 0]) # Forward (Y+)
        elif action == 1: direction = np.array([0, -1, 0]) # Backward (Y-)
        elif action == 2: direction = np.array([-1, 0, 0]) # Left (X-)
        elif action == 3: direction = np.array([1, 0, 0]) # Right (X+)
        elif action == 4: direction = np.array([0, 0, 1]) # Up
        elif action == 5: direction = np.array([0, 0, -1]) # Down
        
        # Apply wind effect (probabilistic)
        if self._wind_vector[2] > 0.5 and random.random() < 0.2:
            wind_push = np.array([int(np.sign(self._wind_vector[0])), int(np.sign(self._wind_vector[1])), 0])
            direction += wind_push

        # Proposed new location
        new_location = self._agent_location + direction
        
        # Boundary checks
        new_location = np.clip(new_location, 0, [self.grid_size-1, self.grid_size-1, self.max_altitude-1])
        
        # Collision checks
        collision = False
        # 1. Building collision
        if self._buildings[new_location[0], new_location[1]] > new_location[2]:
            collision = True
            reward -= 50
            terminated = True
        
        # 2. No-fly zone check
        in_no_fly = False
        for zone in self._no_fly_zones:
            if zone[0] <= new_location[0] <= zone[2] and zone[1] <= new_location[1] <= zone[3]:
                in_no_fly = True
                reward -= 5
        
        if not collision:
            self._agent_location = new_location
            
            # Distance reward - PRIORITIZE HORIZONTAL MOVEMENT
            # Separate horizontal (XY) and vertical (Z) distances
            prev_xy_dist = np.linalg.norm(self._target_location[:2] - prev_location[:2])
            curr_xy_dist = np.linalg.norm(self._target_location[:2] - self._agent_location[:2])
            
            # Reward horizontal progress MUCH more
            reward += (prev_xy_dist - curr_xy_dist) * 5.0  # 5x weight on horizontal
            
            # Small reward for vertical alignment (getting to ground level)
            prev_z_dist = abs(self._target_location[2] - prev_location[2])
            curr_z_dist = abs(self._target_location[2] - self._agent_location[2])
            reward += (prev_z_dist - curr_z_dist) * 0.5  # Only 0.5x weight on vertical
            
            if curr_xy_dist < prev_xy_dist:
                reward += 2.0 # Big bonus for horizontal progress
            else:
                reward -= 1.0 # Penalty for moving away horizontally
 
        # If agent is at target, automatically deliver!
        if np.array_equal(self._agent_location, self._target_location):
            reward += 500 # HUGE reward for delivery
            terminated = True
            
        # Interaction (Action 7) - No longer needed for delivery but kept for compatibility
        if action == 7:
            reward -= 1 # Wasted interaction

        # Battery consumption
        drain = self.BATTERY_DRAIN_HOVER
        if action < 6: # Moving
            drain = self.BATTERY_DRAIN_MOVE * 0.5 # Cheaper to move
            if action == 4: drain += 0.1 # Ascending costs more
        
        if self._has_package:
            drain += self.BATTERY_DRAIN_CARRY
            
        self._battery -= drain
        if self._battery <= 0:
            reward -= 50 # Less penalty for dying (encourage risk)
            terminated = True
            
        # Time penalty
        reward -= 0.05 # Lower time penalty
        
        if self._current_step >= self.MAX_STEPS:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            if self.renderer:
                self.renderer.render_frame()

    def _render_frame(self):
        # Placeholder for rgb_array rendering if needed
        return np.zeros((500, 500, 3), dtype=np.uint8)

    def close(self):
        if self.renderer:
            self.renderer.close()
