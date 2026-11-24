"""Draw the inkatoms wordmark: the 'o' is a benzene ring and the i's dot is an atom (both red).

Everything molecular is drawn by inkatoms itself; the letters are Avenir Next outlines.
    python docs/_static/logo/make_logo.py [--palette ink-red] [--dot atom] [--molecules t,k]
Writes inkatoms_logo{,_dark}.{svg,pdf,png}, a ring-only mark and favicons next to this file.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath

import inkatoms as ia
from inkatoms._elements import DATA
from inkatoms.structure import Structure

HERE = Path(__file__).resolve().parent
FONT = "Avenir Next"
ia.fonts.use_collection_faces((FONT,))
BOLD = FontProperties(family=FONT, weight="bold")
MED = FontProperties(family=FONT, weight="medium")
SIZE = 100.0                       # font size in data units; the x-height is ~50
A_PER_UNIT = 0.1                   # Å per data unit when a molecule is drawn at letter scale
VDW_C = DATA["C"][1]

# ink: "ink" letters, atoms: "atoms" letters, dot: i's dot, ring: the benzene "o"
PALETTES = {
    "ink-red":   dict(ink="#0b1f3a", atoms="#0b1f3a", dot="#d62828", ring="#d62828"),
    "ink":       dict(ink="#0b1f3a", atoms="#2a5d9f", dot="#0b1f3a", ring="#2a5d9f"),
    "teal":      dict(ink="#0b1f3a", atoms="#1b8a7e", dot="#0b1f3a", ring="#1b8a7e"),
    "coral":     dict(ink="#0b1f3a", atoms="#0b1f3a", dot="#e4572e", ring="#e4572e"),
    "jmol":      dict(ink="#1f1f1f", atoms="#1f1f1f", dot="#ff0d0d", ring="#3050f8"),
    "violet":    dict(ink="#2b1b4d", atoms="#6d4fc8", dot="#2b1b4d", ring="#6d4fc8"),
    "okabe-ito": dict(ink="#0b1f3a", atoms="#0072b2", dot="#e69f00", ring="#0072b2"),
    "graphite":  dict(ink="#262626", atoms="#262626", dot="#c8962e", ring="#c8962e"),
    "mono":      dict(ink="#0b1f3a", atoms="#0b1f3a", dot="#0b1f3a", ring="#0b1f3a"),
}
DARK = {"ink": "white", "#0b1f3a": "white", "#1f1f1f": "white", "#262626": "white",
        "#2b1b4d": "white", "#2a5d9f": "#9cc4ec", "#1b8a7e": "#6fd3c4", "#6d4fc8": "#b9a6f2",
        "#0072b2": "#56b4e9", "#d62828": "#ff6b6b"}


def dark(pal):
    return {k: DARK.get(v, v) for k, v in pal.items()}


def flat(color):
    """inkatoms options for a single-colour, edge-free drawing."""
    return dict(colors={"C": color}, edge_colors={"C": color}, atom_edgewidth=0,
                bond_color=color, bond_edgecolor=color, bond_edgewidth=0,
                depth_cue=False, pad=0.0)


def inset(ax, centre, half):
    cx, cy = centre
    sub = ax.inset_axes([cx - half, cy - half, 2 * half, 2 * half], transform=ax.transData)
    sub.patch.set_alpha(0)
    return sub


def square_limits(sub):
    (a, b), (c, d) = sub.get_xlim(), sub.get_ylim()
    h = max(b - a, d - c) / 2
    sub.set_xlim((a + b) / 2 - h, (a + b) / 2 + h)
    sub.set_ylim((c + d) / 2 - h, (c + d) / 2 + h)


def molecule_at(ax, points, bonds, color, r, width, shading="flat"):
    """Draw atoms at `points` (data units) of radius r joined by bonds of the given width."""
    pts = np.asarray(points, float)
    lo, hi = pts.min(0) - r - 1, pts.max(0) + r + 1
    c, half = (lo + hi) / 2, (hi - lo).max() / 2
    sub = inset(ax, c, half)
    s = Structure(["C"] * len(pts), np.c_[(pts - c) * A_PER_UNIT, np.zeros(len(pts))])
    ia.draw(sub, s, view="z", bonds=list(bonds), shading=shading, autoscale=False,
            ball_scale=r * A_PER_UNIT / VDW_C, bond_width=width * A_PER_UNIT, **flat(color))
    h = half * A_PER_UNIT
    sub.set_xlim(-h, h); sub.set_ylim(-h, h)


def draw_ring(ax, centre, half, color, ball=0.24, bond=0.36):
    ang = np.arange(6) * np.pi / 3 + np.pi / 2
    ring = Structure(["C"] * 6, 1.39 * np.c_[np.cos(ang), np.sin(ang), np.zeros(6)])
    sub = inset(ax, centre, half)
    ia.draw(sub, ring, view="z", shading="flat", ball_scale=ball, bond_width=bond, **flat(color))
    square_limits(sub)


def draw_dot(ax, kind, centre, color):
    cx, cy = centre
    if kind == "atom":
        sub = inset(ax, (cx, cy), 9.5)
        ia.draw(sub, Structure(["C"], np.zeros((1, 3))), view="z", shading="soft",
                ball_scale=0.5, **flat(color))
        square_limits(sub)
    elif kind == "diatomic":
        molecule_at(ax, [(cx - 5.5, cy - 2.2), (cx + 5.5, cy + 4.2)], [(0, 1)], color,
                    r=5.6, width=4.5, shading="soft")
    else:
        raise ValueError(kind)


# letters rebuilt as molecules; geometry follows Avenir Next's own glyphs at SIZE=100
def molecule_t(ax, x, color):
    """Medium 't': four bonds out of one atom at the crossbar, like a projected CH4."""
    cx, cy, r, w = x + 14.05, 43.5, 6.0, 8.0
    pts = [(cx, cy), (cx, 61 - r), (x + 0.3 + r, cy), (x + 31.3 - r, cy), (cx, r - 0.8)]
    molecule_at(ax, pts, [(0, 1), (0, 2), (0, 3), (0, 4)], color, r, w)
    return x + 31.3


def molecule_k(ax, x, color):
    """Bold 'k': a stem with two branches off one atom."""
    sx, r, w = x + 14.75, 9.5, 15.0
    j = (sx, 27.0)
    pts = [j, (sx, 75.6 - r), (sx, r), (x + 50.5, 49.8 - r), (x + 51.0, r)]
    molecule_at(ax, pts, [(0, 1), (0, 2), (0, 3), (0, 4)], color, r, w)
    return x + 59.9


def glyph(ax, text, x, prop, color):
    p = TextPath((x, 0), text, size=SIZE, prop=prop)
    ax.add_patch(PathPatch(p, fc=color, ec="none", lw=0))
    return p.get_extents().x1


def advance(text, prop):
    """Horizontal advance of `text`, measured as the shift it gives a following 'n'."""
    w = lambda t: TextPath((0, 0), t, size=SIZE, prop=prop).get_extents().x1
    return w(text + "n") - w("n")


def wordmark(ax, pal, dot="atom", molecules=(), ring_scale=1.12, gap=2.5):
    stem = TextPath((0, 0), "ı", size=SIZE, prop=BOLD).get_extents()
    glyph(ax, "ı", 0.0, BOLD, pal["ink"])                       # dotless i
    x = advance("ı", BOLD)
    if "k" in molecules:
        glyph(ax, "n", x, BOLD, pal["ink"])
        x = molecule_k(ax, x + advance("n", BOLD), pal["ink"])
    else:
        x = glyph(ax, "nk", x, BOLD, pal["ink"])
    x = glyph(ax, "a", x + 1.5, MED, pal["atoms"])
    if "t" in molecules:
        x = molecule_t(ax, x + 4.0, pal["atoms"]) + 1.0
    else:
        x = glyph(ax, "t", x, MED, pal["atoms"])
    o = TextPath((x + gap, 0), "o", size=SIZE, prop=MED).get_extents()
    half = ring_scale * (o.y1 - o.y0) / 2
    draw_ring(ax, ((o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2), half, pal["ring"])
    right = glyph(ax, "ms", o.x1 + gap, MED, pal["atoms"])
    draw_dot(ax, dot, ((stem.x0 + stem.x1) / 2, 66.0), pal["dot"])
    return right


Y0, Y1, PAD = -6.0, 86.0, 2.0


def wordmark_figure(pal, height=1.2, **kw):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    right = wordmark(ax, pal, **kw)
    x0, x1 = -PAD, right + PAD
    ax.set_xlim(x0, x1); ax.set_ylim(Y0, Y1); ax.set_aspect("equal"); ax.axis("off")
    fig.set_size_inches(height * (x1 - x0) / (Y1 - Y0), height)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", default="ink-red", choices=list(PALETTES))
    ap.add_argument("--dot", default="atom", choices=["atom", "diatomic"])
    ap.add_argument("--molecules", default="", help="extra letters drawn as molecules: t, k")
    a = ap.parse_args()
    plt.rcParams["svg.fonttype"] = "path"
    mols = tuple(m for m in a.molecules.split(",") if m)
    pal = PALETTES[a.palette]

    for stem, p in (("inkatoms_logo", pal), ("inkatoms_logo_dark", dark(pal))):
        fig = wordmark_figure(p, dot=a.dot, molecules=mols)
        for ext in ("svg", "pdf", "png"):
            fig.savefig(HERE / f"{stem}.{ext}", dpi=400, transparent=True)
        plt.close(fig)

    # ring-only mark (avatar, favicon): thicker ring so it holds up at 16 px
    for px in (16, 32, 48, 180, 512):
        fig = plt.figure(figsize=(1, 1))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal"); ax.axis("off")
        bold = px <= 48
        draw_ring(ax, (0, 0), 0.98, pal["ring"], ball=0.32 if bold else 0.26,
                  bond=0.5 if bold else 0.38)
        fig.savefig(HERE / f"favicon_{px}.png", dpi=px, transparent=True)
        if px == 512:
            fig.savefig(HERE / "inkatoms_mark.svg", transparent=True)
        plt.close(fig)
    try:
        from PIL import Image
        Image.open(HERE / "favicon_48.png").save(HERE / "favicon.ico",
                                                 sizes=[(16, 16), (32, 32), (48, 48)])
    except ImportError:
        pass


if __name__ == "__main__":
    main()
