"""
Volumetric data behind a structure
==================================

``density_slice`` samples a 3D grid (for example from a cube file via
``inkatoms.read_cube``) on the page plane and draws it under the atoms.
"""
import matplotlib.pyplot as plt
import numpy as np
from ase.build import molecule

import inkatoms as ia

cell = np.diag([8.0, 8.0, 8.0])
g = np.linspace(0, 1, 64, endpoint=False)
X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
orbital = np.exp(-r2 / 0.01) * (X - 0.5) * 40       # a p-like lobe pair
mol = molecule("H2O")
mol.positions += 4.0
fig, ax = plt.subplots(figsize=(3.6, 3.6))
art = ia.draw(ax, mol, view="z", shading="soft")
ax.set_xlim(1.5, 6.5)
ax.set_ylim(1.5, 6.5)
cs = ia.density_slice(art, orbital, cell, cmap="RdBu_r", levels=np.linspace(-1, 1, 15))
plt.show()
