"""
Periodic structures
===================

Ghost images, VESTA-style boundary atoms, lattice planes, cell vectors and an
axis triad.
"""
import matplotlib.pyplot as plt
from ase.build import bulk

import inkatoms as ia

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
cu = bulk("Cu", "fcc", a=3.61, cubic=True)
ia.draw(axes[0], cu, view="z", rotate="-25x,20y", ghosts=1.9, cell=True, shading="soft")
axes[0].set_title("ghosts=1.9")
nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
art = ia.draw(axes[1], nacl, view="z", rotate="-25x,30y", boundary=True, cell=True,
              planes=[(1, 1, 1)], style="vesta", pbc_bonds=False)
ia.axes_triad(axes[1], art)
axes[1].set_title("boundary + (111)")
art = ia.draw(axes[2], cu, view="z", rotate="-25x,20y", cell=True, cell_vectors=True,
              boundary=True, mode="spacefill", radii={"Cu": 0.5})
axes[2].set_title("cell vectors")
plt.show()
