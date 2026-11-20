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

from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import BFGS

# 7. trajectories: a scripted reaction on Pt(111), OH + H -> H2O, which then leaves
slab = fcc111("Pt", size=(4, 4, 3), vacuum=8.0)
top = slab.positions[slab.get_tags() == 1]
z_top = top[:, 2].max()
centre = top[:, :2].mean(axis=0)
near = top[np.argsort(np.linalg.norm(top[:, :2] - centre, axis=1))[:3]]
hollow = near[:, :2].mean(axis=0)
o0 = np.array([*hollow, z_top + 1.2])
ang = np.radians(52)
h1_rel = 0.97 * np.array([np.sin(ang), 0, np.cos(ang)])
h2_rel = 0.97 * np.array([-np.sin(ang), 0, np.cos(ang)])
pt_far = top[np.argmin(np.linalg.norm(top[:, :2] - (hollow + [-2.8, 0]), axis=1))]
h2_start = pt_far + [0, 0, 1.55]

rng = np.random.default_rng(4)
frames = []
n_frames = 90
for f in range(n_frames):
    o = o0 + [0, 0, 0.16 * max(f - 55, 0)]
    s = np.clip((f - 12) / 25, 0, 1)
    lift = [0, 0, 0.6 * np.sin(np.pi * s)]          # over the Pt in between, not onto it
    h2 = (1 - s) * h2_start + s * (o0 + h2_rel) + lift if f < 55 else o + h2_rel
    pos = np.vstack([slab.positions, o, o + h1_rel, h2])
    pos += rng.normal(0, 0.03, pos.shape)
    a = Atoms(list(slab.get_chemical_symbols()) + ["O", "H", "H"], positions=pos, cell=slab.cell, pbc=slab.pbc)
    a.new_array("uid", np.arange(len(a)))
    a.info["time_ps"] = 0.05 * f
    frames.append(a)
O, H1, H2 = len(slab), len(slab) + 1, len(slab) + 2
cut = {"Pt-O": 2.4, "Pt-H": 1.9, "O-H": 1.2, "Pt-Pt": 3.0}
rx = ia.find_reactions(frames, cutoffs=cut, only=[O, H2], min_frames=4)[0]
print(rx)

# relaxation of an O adatom on Pt(111) with EMT, bottom layer fixed
start = fcc111("Pt", size=(4, 4, 3), vacuum=8.0)
add_adsorbate(start, "O", 1.5, "fcc")
start.set_constraint(FixAtoms(indices=[a.index for a in start if a.tag == 3]))
relaxed = start.copy()
relaxed.calc = EMT()
BFGS(relaxed, logfile=None).run(fmax=0.02)

fig = plt.figure(figsize=(9.2, 5.4))
top_row = fig.add_gridspec(1, 4, left=0.01, right=0.99, top=0.9, bottom=0.55, wspace=0.05)
strip = [fig.add_subplot(top_row[0, k]) for k in range(4)]
ia.event_strip(strip, frames, rx, focus=O, view="-y", window=(9, 7.5), anchor=(0.5, 0.35), slab=3.0,
               cutoffs=cut, shading="soft", fontsize=8, lead=6, settle=3)
fig.text(0.01, 0.955, "find_reactions + event_strip", fontsize=9, fontweight="bold")

bottom = fig.add_gridspec(1, 3, left=0.02, right=0.985, top=0.43, bottom=0.08, wspace=0.3,
                          width_ratios=[1, 1.1, 1.15])
ax = fig.add_subplot(bottom[0, 0])
last = frames[-1]
ia.draw(ax, last, view="-y", cutoffs=cut, shading="soft", pbc_bonds=False, autoscale=False,
        highlight={O: "#e8503a", H1: "#f2f2f2", H2: "#f2f2f2"}, highlight_front=True,
        trails={"frames": frames, "ids": [O, H2], "style": "arrow", "color": "#c0392b"})
q = o0 @ ia.view_matrix("-y").T
ax.set_xlim(q[0] - 5.5, q[0] + 5.5)
ax.set_ylim(q[1] - 4.5, q[1] + 6.5)
ax.set_title('trails={"style": "arrow"}', fontsize=8, loc="left")

ax = fig.add_subplot(bottom[0, 1])
ia.plot_neighbours(ax, frames, O, cutoffs=cut, events=rx.events, min_frames=4,
                   colors={"Pt": "#5f6f7a", "H": "#c0392b"})
ax.set_title("plot_neighbours (the O)", fontsize=8, loc="left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

ax = fig.add_subplot(bottom[0, 2])
art, cmp_ = ia.draw_comparison(ax, start, relaxed, view="-y", rotate="15x", pbc_bonds=False,
                               arrow_scale=12, min_shift=0.01, vmax=0.12, shading="soft")
cax = ax.inset_axes([0.25, -0.05, 0.5, 0.035])
cb = fig.colorbar(art.mappable, cax=cax, orientation="horizontal")
cb.set_label("moved (Å)", fontsize=7, labelpad=1)
cb.ax.tick_params(labelsize=6)
ax.set_title("draw_comparison: O on Pt(111) relaxed (EMT)", fontsize=8, loc="left")
save(fig, "dynamics")
