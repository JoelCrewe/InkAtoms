"""
Events, trails and comparisons
==============================

``find_reactions`` finds bonds that form and break in a trajectory (a change
counts once it lasts ``min_frames``), ``event_strip`` draws each step,
``trails=`` shows where atoms went, and ``draw_comparison`` shows how one
structure differs from another, matched and aligned automatically.
"""
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.cluster import Icosahedron
from ase.optimize import BFGS

import inkatoms as ia

# fake trajectory: H2S loses an H, then the SH leaves a surface S
rng = np.random.default_rng(0)
frames = []
for f in range(60):
    h1 = [0.96 + (0.06 * (f - 20) if f > 20 else 0), 0.93, 0]
    sh = [0, 0, 0.12 * max(f - 40, 0)]
    pos = np.array([[0, 0, 0], h1, [-0.96, 0.93, 0], [0, -2.3, 0], [2.3, -2.3, 0]], float)
    pos[[0, 2]] += sh
    pos += rng.normal(0, 0.02, pos.shape)
    a = Atoms("SHHSS", positions=pos)
    a.new_array("uid", np.arange(5))
    a.info["time_ps"] = 0.1 * f
    frames.append(a)
cut = {"S-H": 1.6, "S-S": 2.45}
rx = ia.find_reactions(frames, cutoffs=cut, min_frames=4)[0]
print(rx)

fig = plt.figure(figsize=(9, 5.4))
strip = [fig.add_subplot(2, 4, k + 1) for k in range(4)]
ia.event_strip(strip, frames, rx, around=6, cutoffs=cut, view="z", rotate="-55x", shading="soft")

ax = fig.add_subplot(2, 4, (5, 6))
ia.draw(ax, frames[-1], view="z", rotate="-55x", cutoffs=cut, shading="soft",
        trails={"frames": frames, "ids": [0, 1, 2], "color": "time", "smooth": 3})
ax.set_title("trails (colour = time)", fontsize=8, loc="left")

cl = Icosahedron("Cu", 3)
cl.rattle(0.2, seed=1)
before = cl.copy()
cl.calc = EMT()
BFGS(cl, logfile=None).run(fmax=0.05)
ax = fig.add_subplot(2, 4, (7, 8))
art, comparison = ia.draw_comparison(ax, before, cl, view="z", rotate="-20x", arrow_scale=3,
                                     min_shift=0.05, shading="soft")
ia.colorbar(art, ax=ax, label="moved (Å)")
ax.set_title(f"relaxation, RMSD {comparison.rmsd:.2f} Å", fontsize=8, loc="left")
plt.show()
