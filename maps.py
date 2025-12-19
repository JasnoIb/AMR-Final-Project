import numpy as np
import matplotlib.pyplot as plt

class OccupancyGridMap:
    def __init__(self, x_width, y_width, resolution, safety_margin=0.5):
        self.resolution = resolution
        self.x_width = x_width
        self.y_width = y_width
        self.safety_margin = safety_margin 
        
        # Grid dimensions
        self.width_idx = int(round(x_width / resolution))
        self.height_idx = int(round(y_width / resolution))
        
        # Grid stores FLOAT costs (0.0 = Safe, 100.0 = Occupied)
        self.grid = np.zeros((self.width_idx, self.height_idx), dtype=float)
        self.OBSTACLE_COST = 100.0

    def get_index(self, x, y):
        ix = int(round(x / self.resolution))
        iy = int(round(y / self.resolution))
        ix = np.clip(ix, 0, self.width_idx - 1)
        iy = np.clip(iy, 0, self.height_idx - 1)
        return ix, iy

    def get_cost(self, x, y):
        ix, iy = self.get_index(x, y)
        return self.grid[ix, iy]

    def add_rect_obstacle(self, x, y, width, height):
        min_x = x - self.safety_margin
        max_x = x + width + self.safety_margin
        min_y = y - self.safety_margin
        max_y = y + height + self.safety_margin

        ix_min, iy_min = self.get_index(min_x, min_y)
        ix_max, iy_max = self.get_index(max_x, max_y)

        for i in range(ix_min, ix_max + 1):
            for j in range(iy_min, iy_max + 1):
                px = i * self.resolution
                py = j * self.resolution

                dx = max(x - px, px - (x + width), 0)
                dy = max(y - py, py - (y + height), 0)
                dist = np.hypot(dx, dy)

                if x <= px <= x + width and y <= py <= y + height:
                    self.grid[i, j] = self.OBSTACLE_COST
                
                elif dist <= self.safety_margin:
                    # A. Calculate how far we are into the safety margin (0.0 to 1.0)
                    normalized_dist = dist / self.safety_margin
                    # B. The Tuning Knob (Lower = Larger curve)
                    decay_factor = 1.5
                    # C. The Equation: Cost = 100 * e^(-1.5 * dist)
                    cost = self.OBSTACLE_COST * np.exp(-decay_factor * normalized_dist)
                    # D. Update the grid (keep the highest cost if overlapping)
                    self.grid[i, j] = max(self.grid[i, j], cost)

    def add_circle_obstacle(self, x, y, radius):
        search_r = radius + self.safety_margin
        ix, iy = self.get_index(x, y)
        r_idx = int(round(search_r / self.resolution))

        x_min = max(0, ix - r_idx)
        x_max = min(self.width_idx, ix + r_idx)
        y_min = max(0, iy - r_idx)
        y_max = min(self.height_idx, iy + r_idx)

        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                px = i * self.resolution
                py = j * self.resolution
                
                dist_center = np.hypot(px - x, py - y)
                dist_surface = dist_center - radius

                if dist_surface <= 0:
                    self.grid[i, j] = self.OBSTACLE_COST
                elif dist_surface <= self.safety_margin:
                    # --- UPDATED DECAY FACTOR ---
                    normalized_dist = dist_surface / self.safety_margin
                    decay_factor = 1.5
                    cost = self.OBSTACLE_COST * np.exp(-decay_factor * normalized_dist)
                    
                    self.grid[i, j] = max(self.grid[i, j], cost)

    def is_occupied(self, x, y):
        if x < 0 or x >= self.x_width or y < 0 or y >= self.y_width:
            return True
        return self.get_cost(x, y) >= (self.OBSTACLE_COST - 1.0)

    def plot(self, ax=None):
        if ax is None: fig, ax = plt.subplots()
        ax.imshow(self.grid.T, cmap='Greys', origin='lower', 
                  extent=[0, self.x_width, 0, self.y_width], vmin=0, vmax=100)