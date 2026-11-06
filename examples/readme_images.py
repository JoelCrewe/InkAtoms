"""Render the example gallery into docs/images/.

    python examples/gallery.py

Needs ASE for the example structures (pip install -e ".[dev]").
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk, molecule, mx2

import inkatoms as ia
from inkatoms import select

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# 1. the four presets on DMSO (S, O, C, H) and acetamide (N, O, C, H)
mols = [molecule("C2H6SO"), molecule("CH3CONH2")]
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6))
for col, name in enumerate(["jmol", "cpk", "greyscale", "bw"]):
    for row, mol in enumerate(mols):
        ax = axes[row, col]
        art = ia.draw(ax, mol, view="auto", rotate="25x", style=name)
        ia.legend(ax, art, loc="lower left", fontsize=6, markersize=5,
                  ncol=4, columnspacing=0.6, bbox_to_anchor=(0, -0.12))
        if row == 0:
            ax.set_title(name, fontsize=9)
save(fig, "styles")

# 2. shading and modes
caff = molecule("C4H4S")
fig, axes = plt.subplots(1, 5, figsize=(7.5, 1.8))
for ax, (title, kw) in zip(axes, [("flat", {}), ("glossy", {"shading": "glossy"}),
                                  ("soft", {"shading": "soft"}),
                                  ("spacefill, soft", {"mode": "spacefill", "shading": "soft"}),
                                  ("sticks", {"mode": "sticks", "bond_width": 0.3})]):
    ia.draw(ax, caff, view="auto", rotate="-40x", style="jmol", **kw)
    ax.set_title(title, fontsize=8)
save(fig, "shading")

# 3. depth cueing on a periodic slab: MoS2 tiled, side and top views with the cell
slab = mx2("MoS2", size=(4, 4, 1), vacuum=2.0)
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
ia.draw(axes[0], slab, view="z", cell=True, depth_fade=0.6)
axes[0].set_title("top view, cell, pbc half-bonds")
ia.draw(axes[1], slab, view="z", rotate="-55x,10z", depth_fade=0.75, shading="soft")
axes[1].set_title("tilted, depth fade + soft")
cu = select.around(bulk("Cu", "fcc", a=3.61, cubic=True).repeat((4, 4, 4)),
                   at=None, radius=6.5)
ia.draw(axes[2], cu, view=(1, 1, 1), mode="spacefill", shading="soft", depth_fade=0.7)
axes[2].set_title("Cu cluster, spacefill along [111]")
save(fig, "depth")

# 4. a figure-panel workflow: crop a slab around a site, highlight, annotate
big = select.tile(mx2("MoS2", size=(3, 3, 1), vacuum=4.0), (3, 3, 1))
site = 4                                          # an S atom of the original cell
cut = select.slab(big, axis="y", half_width=3.5, at=[site], keep=[site])
cut = select.box(select.centre(cut, at=[site]), [-9, -np.inf, -6], [9, np.inf, 6], keep=[site])
fig, ax = plt.subplots(figsize=(3.4, 1.9))
art = ia.draw(ax, cut, view="-y", style="jmol", depth_fade=0.7,
              highlight={site: "#e8503a"}, pad=0.8)
x, y = art.xy(site)
ax.annotate("site", xy=(x, y + art.radius(site)), xytext=(x + 3, y + 3.2), fontsize=8,
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333",
                            shrinkA=1, shrinkB=1))
ia.legend(ax, art, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7, markersize=6)
ia.scale_bar(ax, 3, loc="lower left", fontsize=7, bbox_to_anchor=(1.03, 0.05), bbox_transform=ax.transAxes)
save(fig, "panel")

# 5. bond orders, and every molecule drawn at one fixed scale (0.2 in per Å)
names = ["C4H4S", "CH3COOH", "H2CCHCN", "C2H6SO", "C6H6"]
fig, axes = plt.subplots(1, len(names), figsize=(7.5, 1.7))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.85, wspace=0)
for ax, name in zip(axes, names):
    ia.draw(ax, molecule(name), view="auto", bond_orders=True, shading="soft", scale=0.2)
    ax.set_title(name, fontsize=8)
save(fig, "bond_orders")

# 6. features overview for the README
from ase.spacegroup import crystal

rutile = crystal(["Ti", "O"], basis=[(0, 0, 0), (0.3, 0.3, 0)], spacegroup=136,
                 cellpar=[4.59, 4.59, 2.96, 90, 90, 90])
fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.6))
ia.draw(axes[0], select.tile(rutile, (2, 2, 3), centred=False), view="z", rotate="-25x,15y",
        polyhedra={"centers": "Ti"}, style="vesta")
axes[0].set_title("polyhedra", fontsize=9)
pt = __import__("ase.build", fromlist=["fcc111"]).fcc111("Pt", size=(4, 4, 3), vacuum=6.0)
art = ia.draw(axes[1], pt, view="-y", rotate="12x", color_by="z", cmap="cividis", pbc_bonds=False)
axes[1].set_title("color_by", fontsize=9)
m = molecule("CH3COOH")
art = ia.draw(axes[2], m, view="auto", bond_orders=True, shading="soft",
              vectors=np.random.default_rng(3).normal(size=(len(m), 3)) * 0.45)
ia.measure(art, [1, 0, 2])
axes[2].set_title("vectors + measure", fontsize=9)
cu = bulk("Cu", "fcc", a=3.61, cubic=True)
ia.draw(axes[3], cu, view="z", rotate="-25x,20y", ghosts=1.9, cell=True, shading="soft")
axes[3].set_title("ghost images", fontsize=9)
save(fig, "features")
