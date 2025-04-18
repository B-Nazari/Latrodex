import numpy as np
import matplotlib.pyplot as plt
from Latrodex1 import Grid

# Step 1: Get  curves
t = np.linspace(0.01, 0.99*np.pi, 1000)
curve2 = np.array([np.cos(t), np.sin(t)]).T
curve1 = np.array([1.5*np.cos(t), 0.5*np.sin(t)+0.5*t]).T
# Grid and correspondences
grid = Grid(curve1, curve2)
xi, z = grid.compute_correspondences(num_points=50, threshold=1e-3, resolution=0.01, filter_min=0, smooth=True)
pairs = np.column_stack([xi, z])
grid.correspondence_points = {'pairs': pairs, 'xi': pairs[:, 0], 'z': pairs[:, 1]}

# Generate grid
gx, gy = grid.generate_grid_from_correspondence(num_s_values=10)

# Convert to triangulation
tri = grid.to_triangulation(gx, gy)

# Plot with triplot
plt.figure(figsize=(8, 6))
plt.triplot(tri, color='blue', linewidth=0.5)
plt.axis('equal')
plt.axis('off')
plt.title("Triangulated Grid (matplotlib.triplot)")
plt.show()