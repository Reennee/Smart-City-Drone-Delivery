import pygame
import numpy as np

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
        self.font = pygame.font.SysFont("Arial", 18)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BUILDING_COLOR = (100, 100, 100)
        self.NO_FLY_COLOR = (255, 200, 200)

    def render_frame(self):
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill(self.WHITE)
        
        # Draw Grid & Buildings
        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                pygame.draw.rect(canvas, self.GRAY, rect, 1)
                
                # Draw Buildings
                height = self.env._buildings[x, y]
                if height > 0:
                    # Darker shade for taller buildings
                    shade = max(50, 200 - height * 30)
                    color = (shade, shade, shade)
                    pygame.draw.rect(canvas, color, rect)
                    # Draw height text
                    text = self.font.render(f"H:{height}", True, self.WHITE)
                    canvas.blit(text, (rect.centerx - 10, rect.centery - 10))

        # Draw No-Fly Zones
        for zone in self.env._no_fly_zones:
            rect = pygame.Rect(
                zone[0] * self.cell_size,
                zone[1] * self.cell_size,
                (zone[2] - zone[0] + 1) * self.cell_size,
                (zone[3] - zone[1] + 1) * self.cell_size
            )
            s = pygame.Surface((rect.width, rect.height))
            s.set_alpha(100)
            s.fill(self.RED)
            canvas.blit(s, (rect.x, rect.y))

        # Draw Target
        tx, ty, _ = self.env._target_location
        target_center = (
            int((tx + 0.5) * self.cell_size),
            int((ty + 0.5) * self.cell_size)
        )
        pygame.draw.circle(canvas, self.GREEN, target_center, 15)
        
        # Draw Agent
        ax, ay, az = self.env._agent_location
        agent_center = (
            int((ax + 0.5) * self.cell_size),
            int((ay + 0.5) * self.cell_size)
        )
        # Agent color changes with altitude
        agent_color = (0, 0, min(255, 100 + az * 40))
        pygame.draw.circle(canvas, agent_color, agent_center, 20)
        # Draw altitude on agent
        alt_text = self.font.render(f"A:{az}", True, self.WHITE)
        canvas.blit(alt_text, (agent_center[0]-10, agent_center[1]-10))

        # Draw UI Panel
        ui_rect = pygame.Rect(0, self.env.grid_size * self.cell_size, self.width, 100)
        pygame.draw.rect(canvas, self.BLACK, ui_rect)
        
        # Stats
        stats = [
            f"Battery: {self.env._battery:.1f}%",
            f"Steps: {self.env._current_step}",
            f"Altitude: {az}",
            f"Wind: {self.env._wind_vector[2]:.2f}",
            f"Target: {self.env._target_location}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, self.WHITE)
            canvas.blit(text, (20 + (i % 3) * 200, self.height - 90 + (i // 3) * 30))

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.env.metadata["render_fps"])

    def close(self):
        pygame.display.quit()
        pygame.quit()
