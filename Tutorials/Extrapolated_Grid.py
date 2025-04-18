import numpy as np
import matplotlib.pyplot as plt
from Latrodex1 import Grid

# Curves
t = np.linspace(0.01, 0.99*np.pi, 1000)
curve2_raw = np.array([np.cos(t), np.sin(t)]).T
curve1_raw = np.array([1.5*np.cos(t), 2.*np.sin(t)+0.1*t]).T

# Apply spline interpolation to BOTH curves
x1_spline, y1_spline = Grid.centripetal_catmull_rom_spline(curve1_raw[:, 0], curve1_raw[:, 1])
curve1 = np.column_stack([x1_spline, y1_spline])
x2_spline, y2_spline = Grid.centripetal_catmull_rom_spline(curve2_raw[:, 0], curve2_raw[:, 1])
curve2 = np.column_stack([x2_spline, y2_spline])

# Initialize Grid
grid = Grid(curve1=curve1, curve2=curve2)

# Setup plot
fig, axes = plt.subplots(1, 2, figsize=(20, 5))

# Step 1: Grid
xi, z = grid.compute_correspondences(
    num_points=50,
    threshold=1e-3,
    resolution=0.01,
    filter_min=0,
    smooth=False
)
grid.correspondence_points = {'pairs': np.column_stack([xi, z]), 'xi': xi, 'z': z}
gx, gy = grid.generate_grid_from_correspondence(num_s_values=20)
grid.plot_grid(gx, gy, ax=axes[0], title="Grid")

# Step 2: Extrapolate
# Extrapolate in i and j directions
gx, gy = grid.extrapolate_boundary(gx, gy, direction='i', side='min', layers=2)
gx, gy = grid.extrapolate_boundary(gx, gy, direction='j', side='max', layers=2)
gx, gy = grid.extrapolate_boundary(gx, gy, direction='j', side='min', layers=5)
grid.plot_grid(gx, gy,ax=axes[1], title='Extrapolated Grid')
plt.show()
