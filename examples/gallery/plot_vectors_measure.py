"""
Arrows, labels and measurements
===============================

Per-atom vectors (forces, displacements, modes), labels with leader lines, and
true 3D distances, angles and dihedrals drawn onto the projection.
"""
import matplotlib.pyplot as plt
import numpy as np
from ase.build import molecule

import inkatoms as ia

mol = molecule("CH3COOH")
forces = np.random.default_rng(3).normal(size=(len(mol), 3)) * 0.5
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
ia.draw(axes[0], mol, view="auto", vectors=forces, vector_color="#c0392b",
        labels="symbol", label_kw={"offset": (0.45, 0.45), "fontsize": 7})
axes[0].set_title("vectors + labels")
art = ia.draw(axes[1], mol, view="auto", bond_orders=True, shading="soft")
ia.measure(art, [0, 1])          # C=O
ia.measure(art, [1, 0, 2])       # O=C-O
axes[1].set_title("measure()")
plt.show()
