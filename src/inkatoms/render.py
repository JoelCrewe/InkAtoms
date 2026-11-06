"""Draw a structure onto a matplotlib Axes.

Everything (atoms, half-bonds, cell edges, shading) goes into one
PatchCollection sorted back to front, so overlaps are drawn correctly and the
output is a single vector artist however many atoms there are.
"""

from __future__ import annotations

import itertools

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Polygon

from ._elements import DATA
from .bonds import find_bonds
from .structure import Structure, as_structure
from .styles import Style, edge_of, mix, style as get_style
from .view import auto_view, project, view_matrix

MODES = ("ballstick", "spacefill", "sticks")


class StructureArtist:
    """What draw() returns: the collection plus where every atom landed."""

    def __init__(self, ax, collection, structure, projected, radii, R, style):
        self.ax = ax
        self.collection = collection
        self.structure = structure
        self.projected = projected   # (N, 3) page x, page y, depth
        self.radii = radii           # (N,) drawn radius on the page, Å
        self.rotation = R            # rows: page right, up, out in structure coordinates
        self.style = style

    def _rows(self, atom_id):
        rows = self.structure.index_of(atom_id)
        if len(rows) == 0:
            raise KeyError(f"no drawn atom with id {atom_id}")
        return rows

    def xy(self, atom_id):
        """Page position of an atom (the front-most copy if tiled), for annotations."""
        rows = self._rows(atom_id)
        return tuple(self.projected[rows[np.argmax(self.projected[rows, 2])], :2])

    def xy_all(self, atom_id):
        return self.projected[self._rows(atom_id), :2]

    def radius(self, atom_id):
        return float(self.radii[self._rows(atom_id)[0]])

    @property
    def symbols(self):
        """Elements present, heaviest first."""
        syms = set(self.structure.symbols)
        return sorted(syms, key=lambda s: -list(DATA).index(s))

    @property
    def bounds(self):
        """(xmin, xmax, ymin, ymax) of the drawn atoms on the page."""
        x, y = self.projected[:, 0], self.projected[:, 1]
        r = self.radii
        return (float((x - r).min()), float((x + r).max()),
                float((y - r).min()), float((y + r).max()))

    def remove(self):
        self.collection.remove()


def _highlight(s, highlight):
    """-> (mask of highlighted rows, {row: colour or None})"""
    mask = np.zeros(len(s), bool)
    colours = {}
    if highlight is None:
        return mask, colours
    items = highlight.items() if isinstance(highlight, dict) else ((h, None) for h in np.atleast_1d(highlight))
    for atom_id, colour in items:
        rows = s.index_of(atom_id)
        mask[rows] = True
        for r in rows:
            colours[r] = None if colour is None else to_rgb(colour)
    return mask, colours


def _bond_polygon(a, b, half_width):
    d = b - a
    n = np.hypot(*d)
    if n < 1e-9:
        return None
    perp = np.array([-d[1], d[0]]) / n * half_width
    return np.array([a + perp, b + perp, b - perp, a - perp])


def draw(ax, atoms, *, view="z", up=None, rotate=None, style="jmol", mode="ballstick",
         highlight=None, bonds=True, cutoffs=None, bond_scale=1.15, pbc_bonds=None,
         cell=False, radii=None, depth_range=None, autoscale=True, pad=0.5,
         axis_off=True, zorder=2, rasterized=False, **style_changes) -> StructureArtist:
    """Draw a structure on ax.

    atoms       ASE Atoms, pymatgen Structure/Molecule, (symbols, positions) or Structure
    view        direction pointing out of the page: "z" (top), "-y" (side), "x", ... or a
                vector; "auto" turns the structure to its widest face (principal axes)
    up, rotate  see inkatoms.view.view_matrix; rotate="20x,-30y" tilts about the page axes
    style       preset name (jmol, cpk, greyscale, bw) or a Style; extra keyword
                arguments change style fields, e.g. shading="soft", colors={"S": "gold"}
    mode        ballstick | spacefill | sticks
    highlight   atom ids to draw unfaded with a heavier outline, or {id: colour}
    bonds       draw bonds; cutoffs={"Mo-S": 2.85} overrides pair cutoffs (Å), others
                are bond_scale x the sum of covalent radii
    pbc_bonds   bonds across periodic boundaries are drawn as half-bonds towards the
                image (default: follow the structure's pbc)
    cell        draw the unit cell outline
    radii       {symbol: radius in Å} overrides
    depth_range (near, far) depth values for the fade; fix this across panels of a trajectory
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    s = as_structure(atoms)
    st = get_style(style, **style_changes)
    if isinstance(view, str) and view == "auto":
        view = auto_view(s.positions)
    R = view_matrix(view, up=up, rotate=rotate)
    n = len(s)
    P = project(s.positions, R) if n else np.zeros((0, 3))
    bg = np.array(to_rgb(st.background))

    # ---- radii (Å, before depth shrink)
    vdw = np.array([DATA[x][1] for x in s.symbols])
    if mode == "spacefill":
        r3 = vdw.copy()
    elif mode == "sticks":
        r3 = np.full(n, st.bond_width / 2)
    else:
        r3 = st.ball_scale * vdw
    for sym, rad in (radii or {}).items():
        r3[s.symbols == sym] = rad

    # ---- depth cueing
    hl, hl_colours = _highlight(s, highlight)
    depth = P[:, 2]
    if depth_range is not None:
        far, near = min(depth_range), max(depth_range)
    else:
        far, near = (depth.min(), depth.max()) if n else (0.0, 1.0)
    span = near - far

    def closeness(d):
        return np.clip((d - far) / span, 0, 1) if span > 1e-9 else np.ones_like(d)

    t_atom = np.where(hl, 1.0, closeness(depth))
    k_atom = 1.0 - st.depth_shrink * (1.0 - t_atom)
    rr = r3 * k_atom * np.where(hl, st.highlight_scale, 1.0)

    base = np.array([st.color(x) for x in s.symbols]) if n else np.zeros((0, 3))
    for row, colour in hl_colours.items():
        if colour is not None:
            base[row] = colour

    # every drawn piece: (depth, group, order, patch, face rgba, edge rgba, linewidth)
    # group ties pieces of one atom together; bonds/cell (group -1) go before atoms at equal depth
    items = []

    def add(d, group, order, patch, face, edge, lw):
        items.append((d, group, order, patch, face, edge, lw))

    # ---- atoms (+ shading)
    for a in range(n):
        x, y = P[a, :2]
        r = rr[a]
        fade = st.depth_fade * (1 - t_atom[a])
        face = base[a]
        edge = edge_of(face, st.atom_edgecolor, st.darken)
        lw = (st.highlight_edgewidth if hl[a] else st.atom_edgewidth) * k_atom[a]
        if st.shading == "soft":
            body = mix(face, (0, 0, 0), 0.12)
        else:
            body = face
        add(depth[a], a, 0, Circle((x, y), r), (*mix(body, bg, fade), 1.0), (*mix(edge, bg, fade), 1.0), lw)
        if st.shading == "soft":
            steps = 6
            for k in range(1, steps):
                f = k / steps
                rad = r * (1 - 0.82 * f)
                off = (r - rad) * 0.62
                c = mix(face, (1, 1, 1), 0.55 * f ** 1.3) if f > 0.3 else face
                add(depth[a], a, k, Circle((x - off * 0.7071, y + off * 0.7071), rad),
                    (*mix(c, bg, fade), 1.0), (0, 0, 0, 0), 0.0)
        elif st.shading == "glossy":
            for k, (offf, radf, alpha) in enumerate(((0.30, 0.30, 0.30), (0.36, 0.13, 0.75)), start=1):
                add(depth[a], a, k, Circle((x - offf * r, y + offf * r), radf * r),
                    (1, 1, 1, alpha * (1 - fade)), (0, 0, 0, 0), 0.0)
        elif st.shading != "flat":
            raise ValueError("shading must be flat, glossy or soft")

    # ---- bonds
    if bonds and mode != "spacefill" and n > 1:
        bi, bj, shifts = find_bonds(s, cutoffs=cutoffs, scale=bond_scale, pbc=pbc_bonds)
        T = shifts @ s.cell if (s.cell is not None and len(shifts)) else np.zeros((len(bi), 3))
        split = isinstance(st.bond_color, str) and st.bond_color == "split"
        bond_rgb = None if split else np.array(to_rgb(st.bond_color))
        w = st.bond_width
        for i, j, tvec in zip(bi, bj, T):
            pi, pj = s.positions[i], s.positions[j] + tvec
            v = pj - pi
            L = np.linalg.norm(v)
            u = v / L
            si = np.sqrt(max(rr[i] ** 2 - (w / 2) ** 2, 0.0))
            sj = np.sqrt(max(rr[j] ** 2 - (w / 2) ** 2, 0.0))
            if L <= si + sj:
                continue  # spheres overlap: bond hidden
            ai, aj = pi + u * si, pj - u * sj
            mid = 0.5 * (ai + aj)
            if split or tvec.any():
                halves = [(ai, mid, i), (aj - tvec, mid - tvec, j)]
            else:
                halves = [(ai, aj, None)]
            for start, end, owner in halves:
                A, M = start @ R.T, end @ R.T
                dmid = 0.5 * (A[2] + M[2])
                if owner is None:
                    t = 1.0 if (hl[i] and hl[j]) else float(closeness(np.array([dmid]))[0])
                else:
                    t = 1.0 if hl[owner] else float(closeness(np.array([dmid]))[0])
                k = 1.0 - st.depth_shrink * (1.0 - t)
                poly = _bond_polygon(A[:2], M[:2], 0.5 * w * k)
                if poly is None:
                    continue
                face = base[owner] if (split and owner is not None) else (bond_rgb if bond_rgb is not None else base[i])
                edge = edge_of(face, st.bond_edgecolor, st.darken)
                fade = st.depth_fade * (1 - t)
                add(dmid, -1, 0, Polygon(poly, closed=True),
                    (*mix(face, bg, fade), 1.0), (*mix(edge, bg, fade), 1.0), st.bond_edgewidth * k)

    # ---- cell
    corners2d = []
    if cell and s.cell is not None:
        cr = np.array(to_rgb(st.cell_color))
        o = s.origin
        a1, a2, a3 = s.cell
        nseg = 12
        for p0, vec in _cell_edges(o, a1, a2, a3):
            for q in range(nseg):
                A = (p0 + vec * q / nseg) @ R.T
                B = (p0 + vec * (q + 1) / nseg) @ R.T
                dmid = 0.5 * (A[2] + B[2])
                t = float(closeness(np.array([dmid]))[0])
                k = 1.0 - st.depth_shrink * (1.0 - t)
                # overlap segments slightly so no hairline gaps show
                ext = (B[:2] - A[:2]) * 0.02
                poly = _bond_polygon(A[:2] - ext, B[:2] + ext, 0.5 * st.cell_width * k)
                if poly is not None:
                    add(dmid, -2, 0, Polygon(poly, closed=True),
                        (*mix(cr, bg, st.depth_fade * (1 - t)), 1.0), (0, 0, 0, 0), 0.0)
            corners2d.append((p0 @ R.T)[:2])
            corners2d.append(((p0 + vec) @ R.T)[:2])

    # ---- assemble, back to front
    if items:
        d = np.array([it[0] for it in items])
        g = np.array([it[1] for it in items])
        o = np.array([it[2] for it in items])
        order = np.lexsort((o, g, np.round(d, 6)))
        patches = [items[q][3] for q in order]
        coll = PatchCollection(patches, match_original=False, zorder=zorder)
        coll.set_facecolor([items[q][4] for q in order])
        coll.set_edgecolor([items[q][5] for q in order])
        coll.set_linewidth([items[q][6] for q in order])
        coll.set_joinstyle("round")
        coll.set_rasterized(rasterized)
    else:
        coll = PatchCollection([], zorder=zorder)
    ax.add_collection(coll)

    art = StructureArtist(ax, coll, s, P, rr, R, st)

    if autoscale and n:
        x0, x1, y0, y1 = art.bounds
        if corners2d:
            c = np.array(corners2d)
            x0, x1 = min(x0, c[:, 0].min()), max(x1, c[:, 0].max())
            y0, y1 = min(y0, c[:, 1].min()), max(y1, c[:, 1].max())
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    if axis_off:
        ax.set_axis_off()
    return art


def _cell_edges(o, a1, a2, a3):
    """The 12 edges of a parallelepiped as (start, vector)."""
    vecs = (a1, a2, a3)
    edges = []
    for k in range(3):
        others = [vecs[m] for m in range(3) if m != k]
        for c1, c2 in itertools.product((0, 1), repeat=2):
            edges.append((o + c1 * others[0] + c2 * others[1], vecs[k]))
    return edges
