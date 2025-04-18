import numpy as np
from Latrodex1 import Grid

# Step 1: Get  curves
t = np.linspace(0.01, 0.99*np.pi, 1000)
curve2 = np.array([np.cos(t), np.sin(t)]).T
curve1 = np.array([1.5*np.cos(t), 2.*np.sin(t)+0.02]).T

# Step 2: Initialize Grid instance
grid = Grid(curve1, curve2)



# Grid and correspondence
grid = Grid(curve1, curve2)
xi, z = grid.compute_correspondences(num_points=50, threshold=1e-3, resolution=0.01, filter_min=0, smooth=False)
pairs = np.column_stack([xi, z])
grid.correspondence_points = {'pairs': pairs, 'xi': pairs[:, 0], 'z': pairs[:, 1]}



s_intervals = [(0.0, 0.4, 0.15), (0.4, 0.6, 0.01), (0.6, 1.01, 0.05)]
gx, gy = grid.nonuniform_sweep(s_intervals)  

grid.plot_grid(gx, gy, title="Nonuniform Sweep Grid")
