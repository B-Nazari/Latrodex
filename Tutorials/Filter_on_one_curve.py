import numpy as np
import matplotlib.pyplot as plt
from Latrodex1 import Grid

# Curves
t = np.linspace(0.01, 1.35*np.pi, 1000)
curve2_raw = np.array([np.cos(t), np.sin(t)]).T
curve1_raw = np.array([1.5*np.cos(t), 2.2*np.sin(t)+0.25*t]).T

# Apply spline interpolation
x1_spline, y1_spline = Grid.centripetal_catmull_rom_spline(curve1_raw[:, 0], curve1_raw[:, 1])
x2_spline, y2_spline = Grid.centripetal_catmull_rom_spline(curve2_raw[:, 0], curve2_raw[:, 1])
curve1 = np.column_stack([x1_spline, y1_spline])
curve2 = np.column_stack([x2_spline, y2_spline])

# Initialize
grid = Grid(curve1=curve1, curve2=curve2)
fig, axes = plt.subplots(1, 2, figsize=(15, 10))

# Step 1: Raw
xi, z = grid.compute_correspondences(
    num_points=50,
    threshold=1e-3,
    resolution=0.02,
    filter_min=0,
    smooth=False
)
pairs = np.column_stack([xi, z])
grid.correspondence_points = {'pairs': pairs, 'xi': pairs[:, 0], 'z': pairs[:, 1]}
gx, gy = grid.generate_grid_from_correspondence(num_s_values=10)
grid.plot_grid(gx, gy, ax=axes[0], title="Raw", label_parameters=True)

# Step 2: Filter xi
filtered = grid.filter_one_curve(pairs, min_diff=3e-2, param='xi', bounds=(0.3, 0.55))
grid.correspondence_points = {'pairs': filtered, 'xi': filtered[:, 0], 'z': filtered[:, 1]}
gx, gy = grid.generate_grid_from_correspondence(num_s_values=10)
grid.plot_grid(gx, gy, ax=axes[1], title="Filtered xi",label_parameters=True)



plt.tight_layout()
plt.show()
