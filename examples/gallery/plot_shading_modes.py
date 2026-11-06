"""
Shading and drawing modes
=========================

``shading`` is flat, glossy or soft; ``mode`` is ballstick, spacefill or sticks.
All of them stay vector graphics.
"""
import matplotlib.pyplot as plt
from ase.build import molecule

import inkatoms as ia

mol = molecule("C4H4S")
cases = [("flat", {}), ("glossy", {"shading": "glossy"}), ("soft", {"shading": "soft"}),
         ("spacefill", {"mode": "spacefill", "shading": "soft"}), ("sticks", {"mode": "sticks"})]
fig, axes = plt.subplots(1, len(cases), figsize=(9, 1.9))
for ax, (title, kw) in zip(axes, cases):
    ia.draw(ax, mol, view="auto", rotate="-40x", **kw)
    ax.set_title(title, fontsize=9)
plt.show()
