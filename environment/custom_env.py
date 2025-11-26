import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque 

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

        # Action Space
        self.action_space = spaces.Discrete(8)

        # Observation Space
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(15,), dtype=np.float32
        )

        # Internal State
        self._agent_location = np.array([0, 0, 0], dtype=int)
        self._target_location = np.array([0, 0, 0], dtype=int)
        self._battery = 100.0
        self._has_package = False
        self._wind_vector = np.array([0, 0, 0.0], dtype=float)
        self._emergency_active = False
        
        # --- MEMORY SYSTEM (Prevents getting stuck) ---
        self._recent_path = deque(maxlen=20) 
        
        self._buildings = self._generate_buildings()
        self._no_fly_zones = self._generate_no_fly_zones()
        
        # Constants
        self.BATTERY_DRAIN_MOVE = 0.5
        self.BATTERY_DRAIN_HOVER = 0.1
        self.BATTERY_DRAIN_CARRY = 0.2 # Added missing constant
        self.MAX_STEPS = 400 
        self._current_step = 0
        
        # --- REWARD WEIGHTS (The Fix) ---
        self.R_GOAL = 500.0
        self.R_CRASH = -200.0 
        self.R_BATTERY_DEATH = -200.0
        self.R_STEP_PENALTY = -0.5 
        self.R_BACKTRACK = -2.0 
        self.R_PENALTY_NO_FLY = -25.0 # <--- This caused your error!

    def _generate_buildings(self):
        buildings = np.zeros((self.grid_size, self.grid_size), dtype=int)
        num_buildings = 15
        for _ in range(num_buildings):
            x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            if (x, y) != (0, 0): 
                buildings[x, y] = random.randint(1, self.max_altitude - 1)
        return buildings

    def _generate_no_fly_zones(self):
        zones = []
        for _ in range(2):
            x = random.randint(0, self.grid_size-3)
            y = random.randint(0, self.grid_size-3)
            zones.append((x, y, x+2, y+2)) 
        return zones

    def _get_obs(self):
        pos_norm = self._agent_location / [self.grid_size, self.grid_size, self.max_altitude]
        target_vec = (self._target_location - self._agent_location) / self.grid_size
        
        # Simple obstacle sensing
        nearest_obs_dist = 100
        nearest_obs_vec = np.zeros(3)
        
        # Scan immediate surroundings
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    check_pos = self._agent_location + [dx, dy, dz]
                    # Check bounds
                    if (0 <= check_pos[0] < self.grid_size and 
                        0 <= check_pos[1] < self.grid_size and 
                        0 <= check_pos[2] < self.max_altitude):
                        # Check building collision
                        if self._buildings[check_pos[0], check_pos[1]] > check_pos[2]:
                            vec = np.array([dx, dy, dz])
                            dist = np.linalg.norm(vec)
                            if dist < nearest_obs_dist:
                                nearest_obs_dist = dist
                                nearest_obs_vec = vec

        obs = np.concatenate([
            pos_norm,
            [self._battery / 100.0],
            [1.0 if self._has_package else 0.0],
            self._wind_vector,
            target_vec,
            nearest_obs_vec, # Vector to nearest obstacle
            [1.0 if self._emergency_active else 0.0]
        ])
        return obs.astype(np.float32)

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._target_location - self._agent_location),
            "battery": self._battery
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Clear Memory
        self._recent_path.clear()
        
        # 2. Safe Air Drop Start
        while True:
            start_x = random.randint(0, self.grid_size-1)
            start_y = random.randint(0, self.grid_size-1)
            start_z = random.randint(3, self.max_altitude-1) # Start high up
            if self._buildings[start_x, start_y] <= start_z:
                self._agent_location = np.array([start_x, start_y, start_z], dtype=int)
                break

        self._battery = 100.0
        self._has_package = False
        self._current_step = 0
        self._emergency_active = False
        
        # Random target
        while True:
            tx, ty = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            if (tx, ty) != (self._agent_location[0], self._agent_location[1]) and self._buildings[tx, ty] == 0:
                self._target_location = np.array([tx, ty, 0], dtype=int)
                break
                
        self._wind_vector = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)])
        
        if self.render_mode == "human" and self.renderer is None:
            try:
                from environment.rendering import CityDroneRenderer
                self.renderer = CityDroneRenderer(self)
            except ImportError: pass

        return self._get_obs(), self._get_info()

    def step(self, action):
        self._current_step += 1
        reward = 0
        terminated = False
        truncated = False
        
        prev_location = self._agent_location.copy()
        
        # Mapping actions
        direction = np.array([0, 0, 0])
        if action == 0: direction = np.array([0, 1, 0]) 
        elif action == 1: direction = np.array([0, -1, 0]) 
        elif action == 2: direction = np.array([-1, 0, 0]) 
        elif action == 3: direction = np.array([1, 0, 0]) 
        elif action == 4: direction = np.array([0, 0, 1]) 
        elif action == 5: direction = np.array([0, 0, -1]) 

        # Calculate new location
        new_location = self._agent_location + direction
        new_location = np.clip(new_location, 0, [self.grid_size-1, self.grid_size-1, self.max_altitude-1])
        
        # Collision Check
        collision = False
        if self._buildings[new_location[0], new_location[1]] > new_location[2]:
            collision = True
            reward += self.R_CRASH
            terminated = True
        
        # No-Fly Zone Check
        if not collision:
            for zone in self._no_fly_zones:
                if zone[0] <= new_location[0] <= zone[2] and zone[1] <= new_location[1] <= zone[3]:
                    reward += self.R_PENALTY_NO_FLY

        if not collision:
            self._agent_location = new_location
            
            # --- LANDING LOGIC (FIXED: NO FARMING) ---
            
            # Check if we are horizontally aligned with target
            is_over_target = np.array_equal(self._agent_location[:2], self._target_location[:2])
            
            if is_over_target:
                z_dist = self._agent_location[2]
                
                if z_dist == 0:
                    # SUCCESS: Landed!
                    reward += self.R_GOAL
                    terminated = True
                else:
                    # We are above target. 
                    # DO NOT give points for just staying here (prevents farming).
                    
                    if action == 5: 
                        # Big reward ONLY for moving down
                        reward += 25.0 
                    elif action == 4: 
                        # Penalty for going up
                        reward -= 10.0
                    else:
                        # Penalty for hovering/waiting when you should be landing
                        reward -= 5.0 
                        
                    # Do not penalize backtracking here (safe zone)
            else:
                # --- STANDARD FLIGHT ---
                
                # Distance Reward
                dist_prev = np.linalg.norm(self._target_location - prev_location)
                dist_curr = np.linalg.norm(self._target_location - self._agent_location)
                
                # Make moving closer valuable
                reward += (dist_prev - dist_curr) * 10.0 
                
                # Breadcrumb/Memory Penalty (Don't go back to where you just were)
                loc_tuple = tuple(self._agent_location)
                if loc_tuple in self._recent_path:
                    reward += self.R_BACKTRACK 
                else:
                    self._recent_path.append(loc_tuple)
                    
                # Time penalty
                reward += self.R_STEP_PENALTY

        # Battery Logic
        drain = self.BATTERY_DRAIN_HOVER
        if action < 6: drain = self.BATTERY_DRAIN_MOVE
        
        self._battery -= drain 
        if self._battery <= 0:
            reward += self.R_BATTERY_DEATH
            terminated = True
            
        if self._current_step >= self.MAX_STEPS:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human" and self.renderer:
            self.renderer.render_frame()

    def close(self):
        if self.renderer:
            self.renderer.close()
