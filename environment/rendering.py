import pygame
import numpy as np
import math

class CityDroneRenderer:
    def __init__(self, env):
        self.env = env
        self.cell_size = 60
        self.width = env.grid_size * self.cell_size
        self.height = env.grid_size * self.cell_size + 100 # Extra space for UI
        
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 14, bold=True)
        self.title_font = pygame.font.SysFont("Arial", 20, bold=True)
        
        # Colors
        self.bg_color = (30, 30, 35) # Dark slate
        self.grid_color = (50, 50, 60)
        self.building_top = (100, 110, 120)
        self.building_side = (60, 70, 80)
        self.drone_body = (0, 150, 255) # Cyan/Blue
        self.drone_prop = (200, 200, 200)
        self.target_color = (0, 255, 100) # Neon Green
        self.no_fly_color = (255, 50, 50, 50) # Transparent Red
        self.text_color = (240, 240, 240)
        self.WHITE = (255, 255, 255)
        
        self.prop_angle = 0

    def draw_building(self, surface, x, y, size, height):
        # Pseudo-3D effect
        # Draw shadow
        shadow_offset = height * 2
        pygame.draw.rect(surface, (20, 20, 25), 
                        (x + 5, y + 5, size, size))
        
        # Draw base/side (if we were doing full 3D, but top-down is simpler)
        # We'll just make taller buildings brighter/larger appearance or add a "roof" detail
        
        # Main roof
        color_val = min(255, 60 + height * 20)
        color = (color_val, color_val, color_val + 10)
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(surface, color, rect, border_radius=4)
        
        # Roof details (windows/vents)
        pygame.draw.rect(surface, (40, 40, 50), (x+10, y+10, size-20, size-20), 1)
        pygame.draw.circle(surface, (40, 40, 50), (x+size//2, y+size//2), 4)
        
        # Height label
        text = self.font.render(f"{height}F", True, (255, 255, 255))
        surface.blit(text, (x + size//2 - text.get_width()//2, y + size//2 - text.get_height()//2))

    def draw_drone(self, surface, x, y, size, altitude):
        # Pulsing effect based on altitude
        scale = 1.0 + (altitude * 0.1)
        
        # Shadow (smaller and offset based on altitude)
        shadow_offset = 5 + altitude * 2
        shadow_size = int(size * 0.6)
        pygame.draw.ellipse(surface, (0, 0, 0, 100), 
                           (x + size//2 - shadow_size//2 + shadow_offset, 
                            y + size//2 - shadow_size//2 + shadow_offset, 
                            shadow_size, shadow_size))

        # Drone Body (Cross shape)
        center = (x + size//2, y + size//2)
        arm_len = int(size * 0.4 * scale)
        thick = int(6 * scale)
        
        # Arms
        pygame.draw.line(surface, self.drone_body, 
                        (center[0]-arm_len, center[1]-arm_len), 
                        (center[0]+arm_len, center[1]+arm_len), thick)
        pygame.draw.line(surface, self.drone_body, 
                        (center[0]-arm_len, center[1]+arm_len), 
                        (center[0]+arm_len, center[1]-arm_len), thick)
        
        # Center hub
        pygame.draw.circle(surface, (255, 255, 255), center, int(8 * scale))
        
        # Propellers (spinning)
        self.prop_angle = (self.prop_angle + 30) % 360
        prop_dist = arm_len
        prop_pos = [
            (center[0]-prop_dist, center[1]-prop_dist),
            (center[0]+prop_dist, center[1]-prop_dist),
            (center[0]-prop_dist, center[1]+prop_dist),
            (center[0]+prop_dist, center[1]+prop_dist)
        ]
        
        for pos in prop_pos:
            # Draw prop circle
            pygame.draw.circle(surface, (100, 100, 100), pos, int(10 * scale), 1)
            # Draw spinning blade
            blade_len = int(12 * scale)
            angle_rad = math.radians(self.prop_angle)
            dx = math.cos(angle_rad) * blade_len
            dy = math.sin(angle_rad) * blade_len
            pygame.draw.line(surface, self.drone_prop, (pos[0]-dx, pos[1]-dy), (pos[0]+dx, pos[1]+dy), 2)

        # Altitude Text
        alt_text = self.font.render(f"Alt: {altitude}", True, self.target_color)
        surface.blit(alt_text, (x, y - 15))

    def render_frame(self):
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill(self.bg_color)
        
        # Draw Grid
        for x in range(self.env.grid_size + 1):
            pos = x * self.cell_size
            pygame.draw.line(canvas, self.grid_color, (pos, 0), (pos, self.width), 1)
            pygame.draw.line(canvas, self.grid_color, (0, pos), (self.width, pos), 1)

        # Draw No-Fly Zones
        for zone in self.env._no_fly_zones:
            rect = pygame.Rect(
                zone[0] * self.cell_size,
                zone[1] * self.cell_size,
                (zone[2] - zone[0] + 1) * self.cell_size,
                (zone[3] - zone[1] + 1) * self.cell_size
            )
            s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            s.fill(self.no_fly_color)
            # Add hazard stripes
            for i in range(0, rect.width + rect.height, 20):
                pygame.draw.line(s, (255, 0, 0, 100), (i, 0), (0, i), 3)
            canvas.blit(s, (rect.x, rect.y))

        # Draw Buildings
        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                h = self.env._buildings[x, y]
                if h > 0:
                    self.draw_building(canvas, x*self.cell_size, y*self.cell_size, 
                                     self.cell_size-4, h)

        # Draw Target
        tx, ty, _ = self.env._target_location
        t_center = (int((tx + 0.5) * self.cell_size), int((ty + 0.5) * self.cell_size))
        # Pulsing glow
        glow_size = 20 + int(5 * math.sin(pygame.time.get_ticks() * 0.01))
        s = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (0, 255, 100, 50), (glow_size, glow_size), glow_size)
        canvas.blit(s, (t_center[0]-glow_size, t_center[1]-glow_size))
        pygame.draw.circle(canvas, self.target_color, t_center, 10)
        pygame.draw.circle(canvas, self.WHITE, t_center, 12, 2)

        # Draw Agent
        ax, ay, az = self.env._agent_location
        self.draw_drone(canvas, ax*self.cell_size, ay*self.cell_size, self.cell_size, az)

        # Draw UI Panel
        ui_rect = pygame.Rect(0, self.env.grid_size * self.cell_size, self.width, 100)
        pygame.draw.rect(canvas, (20, 20, 25), ui_rect)
        pygame.draw.line(canvas, (0, 255, 255), (0, ui_rect.top), (self.width, ui_rect.top), 2)
        
        # Stats
        stats = [
            (f"BATTERY: {self.env._battery:.1f}%", (0, 255, 255) if self.env._battery > 20 else (255, 50, 50)),
            (f"STEPS: {self.env._current_step}", self.WHITE),
            (f"WIND: {self.env._wind_vector[2]:.2f}", self.WHITE),
            (f"TARGET: {self.env._target_location}", self.target_color)
        ]
        
        for i, (text, color) in enumerate(stats):
            surf = self.title_font.render(text, True, color)
            canvas.blit(surf, (30 + (i % 2) * 300, self.height - 80 + (i // 2) * 35))

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.env.metadata["render_fps"])

    def close(self):
        pygame.display.quit()
        pygame.quit()
