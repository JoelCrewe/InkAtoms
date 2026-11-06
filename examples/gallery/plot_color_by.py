"""
Colour atoms by a value
=======================

Any per-atom number: height, coordination, charges, forces (their length),
or an array of your own, with a matching colorbar.
"""
import matplotlib.pyplot as plt
import numpy as np
from ase.build import fcc111, molecule, add_adsorbate

import inkatoms as ia

slab = fcc111("Pt", size=(4, 4, 3), vacuum=6.0)
add_adsorbate(slab, molecule("CO"), 1.9, "ontop")
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
art = ia.draw(axes[0], slab, view="-y", rotate="10x", color_by="z", cmap="cividis", pbc_bonds=False)
ia.colorbar(art, ax=axes[0], label="height (Å)")

charges = np.random.default_rng(0).normal(0, 0.3, len(slab))
art = ia.draw(axes[1], slab, view="z", color_by=charges, cmap="RdBu_r", vmin=-0.8, vmax=0.8,
              pbc_bonds=False, labels="value", label_kw={"fmt": "{:+.1f}", "fontsize": 5})
ia.colorbar(art, ax=axes[1], label="charge (e)")
plt.show()
