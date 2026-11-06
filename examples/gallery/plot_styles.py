"""
Colour presets
==============

The built-in styles on dimethyl sulfoxide (S, O, C, H).
"""
import matplotlib.pyplot as plt
from ase.build import molecule

import inkatoms as ia

names = ["jmol", "cpk", "vesta", "colourblind", "greyscale", "bw"]
fig, axes = plt.subplots(1, len(names), figsize=(9, 1.9))
for ax, name in zip(axes, names):
    art = ia.draw(ax, molecule("C2H6SO"), view="auto", rotate="25x", style=name)
    ax.set_title(name, fontsize=9)
plt.show()
