import numpy as np
from Latrodex1 import Grid

# Step 1: Get  curves
t = np.linspace(0.01, 0.99*np.pi, 1000)
curve2 = np.array([np.cos(t), np.sin(t)]).T
curve1 = np.array([1.5*np.cos(t), 2.*np.sin(t)+0.02]).T

# Step 2: Initialize Grid instance
grid = Grid(curve1, curve2)



xi_custom = Grid.nonuniform_resolution([
    (0.0, 0.35, 0.05),
    (0.35, 0.5, 0.01),
    (0.5, 1.0, 0.025)
])

# Apply it manually
xi_list = xi_custom
z_list = []
for xi in xi_list:
    x1_xi, y1_xi, x2_z, y2_z, z = grid.find_corresponding_z(xi)
    if z is not None:
        z_list.append(z)

xi_list = xi_list[:len(z_list)]  # match lengths
pairs = np.column_stack([xi_list, z_list])
grid.correspondence_points = {'pairs': pairs, 'xi': pairs[:, 0], 'z': pairs[:, 1]}
gx, gy = grid.generate_grid_from_correspondence(num_s_values=20)

grid.plot_grid(gx, gy, color='b-', linewidth=0.5, title="Nonuniform Resolution Gird")
