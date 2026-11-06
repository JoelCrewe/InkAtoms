"""
Coordination polyhedra
======================

Translucent polyhedra are depth-sorted with the atoms, so the centre atom shows
through and nearer atoms are drawn in front.
"""
import matplotlib.pyplot as plt
from ase.spacegroup import crystal

import inkatoms as ia
from inkatoms import select

rutile = crystal(["Ti", "O"], basis=[(0, 0, 0), (0.3, 0.3, 0)], spacegroup=136,
                 cellpar=[4.59, 4.59, 2.96, 90, 90, 90])
perovskite = crystal(["Sr", "Ti", "O"], basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
                     spacegroup=221, cellpar=[3.905] * 3 + [90] * 3)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
ia.draw(axes[0], select.tile(rutile, (2, 2, 3), centred=False), view="z", rotate="-25x,15y",
        polyhedra={"centers": "Ti"}, style="vesta")
axes[0].set_title("rutile TiO$_2$")
ia.draw(axes[1], select.tile(perovskite, (2, 2, 2), centred=False), view="z", rotate="-30x,25y",
        polyhedra={"centers": "Ti", "alpha": 0.45}, cutoffs={"Sr-O": 0}, pbc_bonds=False,
        boundary=True, cell=True, style="vesta")
axes[1].set_title("SrTiO$_3$, boundary atoms")
plt.show()
