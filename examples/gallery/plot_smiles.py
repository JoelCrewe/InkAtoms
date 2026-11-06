"""
From SMILES, with RDKit bond orders
===================================

``inkatoms.from_smiles`` embeds a molecule in 3D with RDKit and keeps its bond
orders, so double and triple bonds are exact rather than guessed.
"""
import matplotlib.pyplot as plt

import inkatoms as ia

fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
for ax, (name, smi) in zip(axes, [("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
                                  ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
                                  ("acrylonitrile", "C=CC#N")]):
    ia.draw(ax, ia.from_smiles(smi), view="auto", bond_orders=True, shading="soft")
    ax.set_title(name)
plt.show()
