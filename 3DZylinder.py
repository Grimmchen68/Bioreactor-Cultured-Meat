import numpy as np
import pyvista as pv

# Gitter
nx, ny, nz = 50, 50, 50
dx, dy, dz = 1.0/nx, 1.0/ny, 1.0/nz

# Zentrum & Radius
cx, cy = nx//2, ny//2
radius = nx//3

# Zeit (für Rotation)
t = 0.0
omega = 2 * np.pi  # Rotationsgeschwindigkeit

# Maske
mask = np.ones((nx, ny, nz), dtype=bool)

# --- ZYLINDER (Reaktorwand) ---
for i in range(nx):
    for j in range(ny):
        r = np.sqrt((i - cx)**2 + (j - cy)**2)
        for k in range(nz):
            if r > radius:
                mask[i, j, k] = False  # Wand

# --- ZENTRALE WELLE ---
shaft_radius = radius * 0.1
for i in range(nx):
    for j in range(ny):
        r = np.sqrt((i - cx)**2 + (j - cy)**2)
        if r < shaft_radius:
            mask[i, j, :] = False

# --- IMPELLER (rotierend) ---
impeller_z = nz // 2
blade_length = radius * 0.7

for i in range(nx):
    for j in range(ny):
        # Koordinaten relativ zum Zentrum
        x_rel = i - cx
        y_rel = j - cy

        # Rotation
        angle = omega * t
        x_rot = x_rel * np.cos(angle) - y_rel * np.sin(angle)
        y_rot = x_rel * np.sin(angle) + y_rel * np.cos(angle)

        # 2 Blätter (Kreuzform)
        if (abs(y_rot) < 1 and abs(x_rot) < blade_length) or \
           (abs(x_rot) < 1 and abs(y_rot) < blade_length):
            mask[i, j, impeller_z] = False

# --- PyVista Grid ---
grid = pv.ImageData()
grid.dimensions = np.array(mask.shape) + 1
grid.spacing = (dx, dy, dz)
grid.origin = (0, 0, 0)

grid["solid"] = (~mask).astype(int).flatten()

grid = grid.cell_data_to_point_data()
# Oberfläche extrahieren
contours = grid.contour(isosurfaces=[0.5], scalars="solid")

plotter = pv.Plotter()
plotter.add_mesh(contours, color="lightblue", opacity=1.0)
plotter.add_axes()
plotter.show()