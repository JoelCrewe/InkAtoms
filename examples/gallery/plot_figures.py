"""
Structures inside plots, and trajectory strips
==============================================

``inset`` puts a drawing at a data point; ``panels`` draws frames with one
view, one depth fade and identical framing.
"""
import matplotlib.pyplot as plt
import numpy as np
from ase.build import molecule

import inkatoms as ia

fig = plt.figure(figsize=(9, 5))
ax = fig.add_axes([0.08, 0.45, 0.88, 0.5])
x = np.linspace(0, 1, 200)
ax.plot(x, 1.2 * np.exp(-((x - 0.5) / 0.15) ** 2), color="#333333")
ax.set(xlabel="reaction coordinate", ylabel="E (eV)", ylim=(-0.2, 2.0))
mol = molecule("CH3CH2OH")
for xi in (0.1, 0.5, 0.9):
    m = mol.copy()
    m.positions[2] += [0.6 * xi, 0, 0]
    ia.inset(ax, m, (xi, 1.2 * np.exp(-((xi - 0.5) / 0.15) ** 2) + 0.5), size=(0.18, 0.4), view="auto")

frames = []
for k in range(5):
    m = molecule("H2O")
    m.positions[1] += [0.25 * k, 0.1 * k, 0]
    frames.append(m)
axes = [fig.add_axes([0.02 + 0.195 * k, 0.03, 0.18, 0.3]) for k in range(5)]
ia.panels(axes, frames, view="z", highlight=[1])
plt.show()
