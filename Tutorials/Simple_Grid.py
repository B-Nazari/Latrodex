import numpy as np
from Latrodex1 import Grid

# Step 1: Get  curves
t = np.linspace(0.01, 0.99*np.pi, 1000)
curve2 = np.array([np.cos(t), np.sin(t)]).T
curve1 = np.array([1.5*np.cos(t), 2.2*np.sin(t)+0.02]).T

# Step 2: Initiate
grid = Grid(curve1, curve2)

# Step 3: Find Pairs
xi, z = grid.compute_correspondences(num_points=40, smooth=False, filter_min=0, resolution=0.01, threshold=1e-3)
pairs = np.column_stack([xi, z])
grid.correspondence_points = {'pairs': pairs, 'xi': pairs[:, 0], 'z': pairs[:, 1]}

# Step 4: Create Grid

gx, gy = grid.generate_grid_from_correspondence(num_s_values=10)

# Step 5: Plot Grid
grid.plot_grid(gx, gy, title="Simple Grid")

