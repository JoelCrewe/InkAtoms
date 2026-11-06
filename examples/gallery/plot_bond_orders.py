"""
Bond orders at one fixed scale
==============================

``bond_orders=True`` guesses double and triple bonds from bond lengths;
``scale=`` fixes inches per Å so every panel is drawn at the same size.
"""
import matplotlib.pyplot as plt
from ase.build import molecule

import inkatoms as ia

names = ["C4H4S", "CH3COOH", "H2CCHCN", "C2H6SO", "C6H6"]
fig, axes = plt.subplots(1, len(names), figsize=(9, 2.0))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.85, wspace=0)
for ax, name in zip(axes, names):
    ia.draw(ax, molecule(name), view="auto", bond_orders=True, shading="soft", scale=0.22)
    ax.set_title(name, fontsize=9)
plt.show()
