"""Draw a structure onto a matplotlib Axes.

Everything (atoms, half-bonds, polyhedra, rings, arrows, cell edges, planes,
shading) becomes depth-tagged pieces of one scene, sorted back to front into a
single PatchCollection, so overlaps are drawn correctly and the output is one
vector artist however many atoms there are.
"""

from __future__ import annotations

import itertools

import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection, PathCollection
from matplotlib.colors import Colormap, Normalize, to_rgb
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.spatial import ConvexHull, cKDTree

from . import select as _select
from ._elements import DATA
from .bonds import NONMETALS, coordination, find_bonds, find_hbonds, find_rings
from .bonds import bond_orders as guess_orders
from .structure import as_structure
from .styles import edge_of, mix
from .styles import style as get_style
from .view import auto_view, view_matrix

MODES = ("ballstick", "spacefill", "sticks")
AUTO_RASTERIZE = 40000   # rasterize="auto" switches on above this many drawn pieces
SOLID = "solid"
DASHED = (0, (2.0, 1.6))


# ---------------------------------------------------------------- raw geometry
# Paths are built directly (not as Patch objects, which are slow to create);
# editable=True wraps them in PathPatches afterwards.

_UNIT_CIRCLE = Path.circle((0.0, 0.0), 1.0)
_CIRCLE_V, _CIRCLE_C = _UNIT_CIRCLE.vertices, _UNIT_CIRCLE.codes


def _circle(centre, radius):
    return Path(_CIRCLE_V * radius + np.asarray(centre, float), _CIRCLE_C)


def _poly(verts):
    v = np.asarray(verts, float)
    v = np.vstack([v, v[:1]])
    codes = np.full(len(v), Path.LINETO, dtype=Path.code_type)
    codes[0], codes[-1] = Path.MOVETO, Path.CLOSEPOLY
    return Path(v, codes)


def _wedge(centre, radius, theta1, theta2):
    w = Path.wedge(theta1, theta2)
    return Path(w.vertices * radius + np.asarray(centre, float), w.codes)


class StructureArtist:
    """What draw() returns: the drawn pieces plus where every atom landed."""

    def __init__(self, ax, collection, structure, projected, radii, R, style, scene):
        self.ax = ax
        self.collection = collection        # PatchCollection (None when editable=True)
        self.patches = []                   # individual patches when editable=True
        self.structure = structure          # what was drawn (after images/ghosts)
        self.projected = projected          # (N, 3) page x, page y, depth
        self.radii = radii                  # (N,) drawn radius on the page, Å
        self.rotation = R                   # rows: page right, up, out in structure coordinates
        self.style = style
        self.mappable = None                # ScalarMappable for colorbar() when color_by= was used
        self.values = None                  # the per-atom values behind color_by
        self.labels = []                    # Text artists from labels=
        self.extras = []                    # other artists added by draw (cell-vector labels, ...)
        self._scene = scene

    def _rows(self, atom_id):
        rows = self.structure.index_of(atom_id)
        if len(rows) == 0:
            raise KeyError(f"no drawn atom with id {atom_id}")
        return rows

    def row(self, atom_id):
        """Row of an atom (the front-most copy if there are several)."""
        rows = self._rows(atom_id)
        return int(rows[np.argmax(self.projected[rows, 2])])

    def xy(self, atom_id):
        """Page position of an atom (the front-most copy), for annotations."""
        return tuple(self.projected[self.row(atom_id), :2])

    def xy_all(self, atom_id):
        return self.projected[self._rows(atom_id), :2]

    def radius(self, atom_id):
        return float(self.radii[self.row(atom_id)])

    def position(self, atom_id):
        """3D position (Å) of an atom (front-most copy)."""
        return self.structure.positions[self.row(atom_id)]

    def project(self, points):
        """Page (x, y) of 3D points under this drawing's view."""
        xy, _, _ = self._scene.project(np.atleast_2d(points))
        return xy

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
        if self.collection is not None:
            self.collection.remove()
        for p in self.patches + self.labels + self.extras:
            p.remove()


# ---------------------------------------------------------------- scene

class _Scene:
    """Projection, depth cueing and the list of pieces to draw."""

    def __init__(self, positions, R, st, perspective, depth_range, bg):
        self.R, self.st, self.bg = R, st, bg
        n = len(positions)
        self.centre = positions.mean(axis=0) if n else np.zeros(3)
        self.c_page = self.centre @ R.T
        self.D = None if perspective is None else float(perspective)
        depth = (positions @ R.T)[:, 2] if n else np.zeros(1)
        if depth_range is not None:
            self.far, self.near = min(depth_range), max(depth_range)
        else:
            self.far, self.near = float(depth.min()), float(depth.max())
        self.items = []
        self.extent = []

    def project(self, pts):
        q = np.atleast_2d(pts) @ self.R.T
        if self.D is None:
            return q[:, :2], q[:, 2], np.ones(len(q))
        dz = q[:, 2] - self.c_page[2]
        f = self.D / np.maximum(self.D - dz, 0.05 * self.D)
        xy = self.c_page[:2] + (q[:, :2] - self.c_page[:2]) * f[:, None]
        return xy, q[:, 2], f

    def t(self, depth):
        span = self.near - self.far
        d = np.asarray(depth, float)
        return np.clip((d - self.far) / span, 0, 1) if span > 1e-9 else np.ones_like(d)

    def k(self, t):
        return 1.0 - self.st.depth_shrink * (1.0 - np.asarray(t, float))

    def fade(self, t, extra=0.0):
        f = self.st.depth_fade * (1.0 - np.asarray(t, float))
        return f + (1.0 - f) * extra

    def rgba(self, rgb, fade, alpha=1.0):
        return (*mix(rgb, self.bg, fade), alpha)

    def add(self, depth, group, order, patch, face, edge, lw, ls=SOLID, gid=None):
        self.items.append((float(depth), group, order, patch, face, edge, lw, ls, gid))


def _bond_polygon(a, b, half_width):
    d = b - a
    n = np.hypot(*d)
    if n < 1e-9:
        return None
    perp = np.array([-d[1], d[0]]) / n * half_width
    return np.array([a + perp, b + perp, b - perp, a - perp])


def _dashes(a, b, dash, gap):
    """Split the 3D segment a-b into dash pieces."""
    L = np.linalg.norm(b - a)
    if L < 1e-9:
        return []
    u = (b - a) / L
    out, pos = [], 0.0
    while pos < L:
        end = min(pos + dash, L)
        out.append((a + u * pos, a + u * end))
        pos = end + gap
    return out


# (offset across the bond, line width, dashed), in units of style.bond_width
_MULTI = {
    1: [(0.0, 1.0, False)],
    1.5: [(-0.66, 0.52, False), (0.66, 0.52, True)],
    2: [(-0.66, 0.52, False), (0.66, 0.52, False)],
    3: [(-0.86, 0.40, False), (0.0, 0.40, False), (0.86, 0.40, False)],
}

_NAMED_BOND_STYLES = {
    "dashed": {"dashed": True},
    "breaking": {"dashed": True},
    "forming": {"dashed": True, "width": 0.6},
    "hidden": {"hidden": True},
}


def _resolve_orders(s, bond_orders, bonds):
    i, j, _ = bonds
    if bond_orders is None or bond_orders is False:
        return np.ones(len(i), float)
    if bond_orders is True or bond_orders == "auto":
        if s.bond_orders:   # known orders (e.g. RDKit) win over the guess
            table = {frozenset(k): v for k, v in s.bond_orders.items()}
            guess = guess_orders(s, bonds)
            return np.array([table.get(frozenset((s.ids[a], s.ids[b])), g)
                             for a, b, g in zip(i, j, guess)], float)
        return guess_orders(s, bonds).astype(float)
    if isinstance(bond_orders, dict):
        table = {frozenset(k): v for k, v in bond_orders.items()}
        return np.array([table.get(frozenset((s.ids[a], s.ids[b])), 1) for a, b in zip(i, j)], float)
    raise ValueError("bond_orders must be False, True/'auto' or {(id_a, id_b): order}")


def _highlight(s, highlight):
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


def atom_values(s, color_by, *, cutoffs=None, bond_scale=1.15):
    """Per-atom numbers for color_by: "z"/"height", "x", "y", "coordination",
    the name of a per-atom array (vectors give their length), or an array."""
    if color_by is None:
        return None
    if isinstance(color_by, str):
        key = color_by.lower()
        if key in ("z", "height"):
            return s.positions[:, 2].astype(float)
        if key in ("x", "y"):
            return s.positions[:, "xy".index(key)].astype(float)
        if key in ("coordination", "cn"):
            return coordination(s, cutoffs=cutoffs, scale=bond_scale).astype(float)
        if color_by not in s.arrays:
            raise KeyError(f"no per-atom array {color_by!r}; have {sorted(s.arrays)} "
                           "or use 'z', 'coordination'")
        v = np.asarray(s.arrays[color_by], float)
    else:
        v = np.asarray(color_by, float)
    if v.ndim == 2:
        v = np.linalg.norm(v, axis=1)
    if len(v) != len(s):
        raise ValueError(f"color_by has {len(v)} values for {len(s)} atoms")
    return v


def _occlusion(s):
    """0 (exposed) .. 1 (buried) from the number of atoms within 3.6 Å."""
    if len(s) < 3:
        return np.zeros(len(s))
    tree = cKDTree(s.positions)
    counts = np.array([len(x) - 1 for x in tree.query_ball_point(s.positions, 3.6)], float)
    span = np.ptp(counts)
    return (counts - counts.min()) / span if span > 0 else np.zeros(len(s))


def _hull_faces(pts):
    """Faces of the convex hull of pts as ordered vertex-index arrays,
    with coplanar triangles merged; a flat set gives one polygon."""
    def order(verts, normal):
        c = pts[verts].mean(axis=0)
        e1 = np.cross(normal, [1.0, 0, 0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(normal, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        d = pts[verts] - c
        return np.asarray(verts)[np.argsort(np.arctan2(d @ e2, d @ e1))]

    try:
        hull = ConvexHull(pts)
    except Exception:
        c = pts - pts.mean(axis=0)
        normal = np.linalg.svd(c)[2][-1]
        return [order(np.arange(len(pts)), normal)]
    groups = {}
    for simplex, eq in zip(hull.simplices, hull.equations):
        key = tuple(np.round(eq, 2))
        groups.setdefault(key, (eq[:3], set()))[1].update(simplex.tolist())
    return [order(sorted(verts), normal) for normal, verts in groups.values()]


def _arrow(scene, p0, p1, width, colour, fade, group, order, depth=None, gid=None):
    xy, d, f = scene.project(np.array([p0, p1]))
    a, b = xy
    vec = b - a
    L = np.hypot(*vec)
    if L < 1e-6:
        return
    u = vec / L
    t = scene.t(d.mean()) if depth is None else 1.0
    k = float(scene.k(t)) * float(f.mean())
    w = width * k
    head_l, head_w = min(4.0 * w, 0.6 * L), 3.0 * w
    neck = b - u * head_l
    perp = np.array([-u[1], u[0]])
    dd = d.mean() if depth is None else depth
    face = scene.rgba(colour, fade)
    if np.hypot(*(neck - a)) > 1e-6:
        scene.add(dd, group, order, _poly(_bond_polygon(a, neck, 0.5 * w)),
                  face, (0, 0, 0, 0), 0.0, gid=gid)
    scene.add(dd, group, order + 1, _poly([b, neck + perp * head_w / 2, neck - perp * head_w / 2]),
              face, (0, 0, 0, 0), 0.0, gid=gid)
    scene.extent.extend([a, b])


def _cell_edges(o, a1, a2, a3):
    """The 12 edges of a parallelepiped as (start, vector)."""
    vecs = (a1, a2, a3)
    edges = []
    for k in range(3):
        others = [vecs[m] for m in range(3) if m != k]
        for c1, c2 in itertools.product((0, 1), repeat=2):
            edges.append((o + c1 * others[0] + c2 * others[1], vecs[k]))
    return edges


def _plane_polygon(cell, origin, hkl, offset=1.0):
    """Corners (Å) of the lattice plane h u + k v + l w = offset inside the cell."""
    hkl = np.asarray(hkl, float)
    corners = np.array(list(itertools.product((0, 1), repeat=3)), float)
    pts = []
    for a, b in itertools.combinations(range(8), 2):
        if np.abs(corners[a] - corners[b]).sum() != 1:
            continue  # only the 12 cube edges
        fa, fb = corners[a] @ hkl - offset, corners[b] @ hkl - offset
        if abs(fa) < 1e-9:
            pts.append(corners[a])
        if fa * fb < 0:
            pts.append(corners[a] + (corners[b] - corners[a]) * fa / (fa - fb))
    if len(pts) < 3:
        return None
    pts = np.unique(np.round(pts, 9), axis=0)
    cart = origin + pts @ cell
    normal = hkl @ np.linalg.inv(cell).T
    normal /= np.linalg.norm(normal)
    return cart[_hull_faces(cart)[0]] if len(cart) > 3 else cart


# ---------------------------------------------------------------- draw

def draw(ax, atoms, *, view="z", up=None, rotate=None, perspective=None,
         style="jmol", mode="ballstick",
         color_by=None, cmap="viridis", vmin=None, vmax=None, norm=None,
         highlight=None, highlight_front=False,
         bonds=True, cutoffs=None, bond_scale=1.15, bond_orders=False, pbc_bonds=None,
         bond_styles=None, contacts=None,
         vectors=None, vector_scale=1.0, vector_color="#c0392b", vector_width=0.07,
         polyhedra=None, rings=None,
         labels=None, label_kw=None, constraints=None, vacancies=None,
         ghosts=None, boundary=False,
         cell=False, cell_vectors=False, planes=None,
         radii=None, depth_range=None, autoscale=True, pad=0.5, axis_off=True,
         zorder=2, rasterized="auto", editable=False, scale=None,
         **style_changes) -> StructureArtist:
    """Draw a structure on ax and return a StructureArtist.

    Structure and view
      atoms        ASE Atoms, pymatgen Structure/Molecule, RDKit Mol, MDAnalysis
                   Universe, (symbols, positions) or Structure
      view         direction out of the page: "z" (top), "-y" (side), ..., a vector,
                   a 3x3 matrix, or "auto" (principal axes, widest face)
      up, rotate   see view_matrix; rotate="20x,-30y" tilts about the page axes
      perspective  camera distance (Å) from the structure centre; None is orthographic

    Look
      style        jmol | cpk | vesta | colourblind | greyscale | bw | dark, or a Style;
                   extra keyword arguments change style fields (shading="soft", ...)
      mode         ballstick | spacefill | sticks
      color_by     colour atoms by a value: "z", "coordination", an array name
                   ("charges", "forces", ...) or an array; cmap, vmin, vmax, norm as
                   in matplotlib; then inkatoms.colorbar(artist)
      highlight    ids drawn unfaded with a heavier outline, or {id: colour};
                   highlight_front=True keeps them on top of everything

    Bonds
      bonds, cutoffs, bond_scale   cutoffs={"Mo-S": 2.85} overrides pairs (Å)
      bond_orders  True guesses double/triple bonds (or uses known RDKit orders),
                   or {(id_a, id_b): order}; order 1.5 is drawn as solid + dashed
      bond_styles  {(id_a, id_b): "breaking" | "forming" | "dashed" | "hidden" or
                   dict(color=, width=, dashed=, hidden=)}
      contacts     extra dashed contacts: "hbonds" or [(id_a, id_b), ...]
      pbc_bonds    draw half-bonds across periodic boundaries (default: follow pbc)

    Extras
      vectors      per-atom arrows: array (N, 3) or an array name ("forces"),
                   scaled by vector_scale; vector_color ("atom" for element colour)
      polyhedra    True (metal-centred) or dict(centers=, ligands=, alpha=, color=,
                   edgecolor=, hide_bonds=, min_ligands=)
      rings        True or dict(color=, alpha=, aromatic="auto"|True|False, max_size=)
      labels       "symbol" | "id" | "index" | "value" | list | {id: text}; label_kw
                   passed to ax.text (fontsize, color="auto", offset=(dx, dy), ...)
      constraints  True (ASE FixAtoms) or ids: a cross on fixed atoms
      vacancies    (M, 3) positions drawn as dashed empty circles
      ghosts       width (Å): periodic images outside the cell drawn faded
      boundary     repeat atoms on cell faces/edges/corners (as VESTA does)
      cell, cell_vectors, planes   cell outline; labelled a/b/c arrows; lattice
                   planes [(1, 1, 1), {"hkl": (0, 0, 1), "offset": 0.5, "color": ...}]

    Output
      radii        {symbol: radius in Å}; depth_range=(far, near) fixes the fade
      scale        inches per Å (see set_scale)
      rasterized   True / False / "auto" (rasterize very large drawings)
      editable     one patch per piece with gids like "atom-12" (for Illustrator)
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    s = as_structure(atoms)
    st = get_style(style, **style_changes)

    # ---- periodic images
    if ghosts or boundary:
        if s.cell is None:
            raise ValueError("ghosts/boundary need a structure with a unit cell")
        s = _select.images(s, width=float(ghosts) if ghosts else 0.0)
        pbc_bonds = False
    is_image = np.asarray(s.arrays.get("_image", np.zeros(len(s), bool)), bool)
    ghost_fade = 0.6 if ghosts else 0.0
    extra_fade = np.where(is_image, ghost_fade, 0.0)

    if isinstance(view, str) and view == "auto":
        view = auto_view(s.positions)
    R = view_matrix(view, up=up, rotate=rotate)
    n = len(s)
    bg = np.array(to_rgb(st.background))
    scene = _Scene(s.positions, R, st, perspective, depth_range, bg)

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

    hl, hl_colours = _highlight(s, highlight)
    xy, depth, f_atom = scene.project(s.positions) if n else (np.zeros((0, 2)), np.zeros(0), np.zeros(0))
    t_atom = np.where(hl, 1.0, scene.t(depth))
    k_atom = scene.k(t_atom)
    r_eff = r3 * k_atom * np.where(hl, st.highlight_scale, 1.0)   # 3D radius used for clipping
    rr = r_eff * f_atom                                           # drawn radius on the page
    fade_atom = scene.fade(t_atom, extra_fade)

    # ---- colours
    base = np.array([st.color(x) for x in s.symbols]) if n else np.zeros((0, 3))
    values = atom_values(s, color_by, cutoffs=cutoffs, bond_scale=bond_scale)
    mappable = None
    if values is not None:
        cm = colormaps[cmap] if isinstance(cmap, str) else cmap
        if not isinstance(cm, Colormap):
            raise TypeError("cmap must be a colormap or its name")
        finite = np.isfinite(values)
        if norm is None:
            lo = np.nanmin(values) if vmin is None else vmin
            hi = np.nanmax(values) if vmax is None else vmax
            norm = Normalize(lo, hi if hi > lo else lo + 1e-9)
        base[finite] = cm(norm(values[finite]))[:, :3]
        mappable = ScalarMappable(norm=norm, cmap=cm)
        mappable.set_array(values)
    for row, colour in hl_colours.items():
        if colour is not None:
            base[row] = colour
    if st.occlusion > 0 and n:
        base = mix(base, (0, 0, 0), st.occlusion * 0.55 * _occlusion(s))

    # ---- atoms (+ shading, partial occupancy, constraint marks)
    fixed = np.zeros(n, bool)
    if constraints is True:
        fixed = np.asarray(s.arrays.get("fixed", fixed), bool)
    elif constraints is not None:
        fixed = np.isin(s.ids, np.atleast_1d(constraints))
    occ = s.occupancy or (None,) * n

    for a in range(n):
        x, y = xy[a]
        r = rr[a]
        fade = fade_atom[a]
        face = base[a]
        edge_rule = st.atom_edgecolor
        if hl[a] and st.highlight_edgecolor is not None:
            edge_rule = st.highlight_edgecolor
        edge = edge_of(face, edge_rule, st.darken)
        lw = (st.highlight_edgewidth if hl[a] else st.atom_edgewidth) * k_atom[a]
        body = mix(face, (0, 0, 0), 0.12) if st.shading == "soft" else face
        da = depth[a] + (1e6 if (highlight_front and hl[a]) else 0.0)
        gid = f"atom-{s.ids[a]}"
        scene.add(da, a, 0, _circle((x, y), r), scene.rgba(body, fade), scene.rgba(edge, fade), lw, gid=gid)
        pie = occ[a]
        if pie:
            theta = 90.0
            for sym, frac in pie:
                c = to_rgb(st.colors.get(sym) or st.color(sym))
                scene.add(da, a, 1, _wedge((x, y), r * 0.985, theta - 360 * frac, theta),
                          scene.rgba(c, fade), (0, 0, 0, 0), 0.0, gid=gid)
                theta -= 360 * frac
            if theta > 90.0 - 359.9:   # vacancy share
                scene.add(da, a, 1, _wedge((x, y), r * 0.985, 90.0 - 360.0, theta),
                          scene.rgba((1, 1, 1), fade), (0, 0, 0, 0), 0.0, gid=gid)
            scene.add(da, a, 40, _circle((x, y), r), (0, 0, 0, 0), scene.rgba(edge, fade), lw, gid=gid)
        elif st.shading == "soft":
            steps = 6
            for k in range(1, steps):
                fk = k / steps
                rad = r * (1 - 0.82 * fk)
                off = (r - rad) * 0.62
                c = mix(face, (1, 1, 1), 0.55 * fk ** 1.3) if fk > 0.3 else face
                scene.add(da, a, k, _circle((x - off * 0.7071, y + off * 0.7071), rad),
                          scene.rgba(c, fade), (0, 0, 0, 0), 0.0, gid=gid)
        if st.shading == "glossy":
            for k, (offf, radf, alpha) in enumerate(((0.30, 0.30, 0.30), (0.36, 0.13, 0.75)), start=20):
                scene.add(da, a, k, _circle((x - offf * r, y + offf * r), radf * r),
                          (1, 1, 1, alpha * (1 - fade)), (0, 0, 0, 0), 0.0, gid=gid)
        elif st.shading not in ("flat", "soft"):
            raise ValueError("shading must be flat, glossy or soft")
        if fixed[a]:
            for ang in (45.0, -45.0):
                u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]) * 0.62 * r
                scene.add(da, a, 70, _poly(_bond_polygon(np.array([x, y]) - u, np.array([x, y]) + u, 0.07 * r)), scene.rgba((0.1, 0.1, 0.1), fade, 0.85),
                          (0, 0, 0, 0), 0.0, gid=gid)

    # ---- bonds (also needed for polyhedra, rings, h-bonds)
    need_bonds = (bonds and mode != "spacefill") or polyhedra or rings or contacts == "hbonds"
    bond_list = find_bonds(s, cutoffs=cutoffs, scale=bond_scale, pbc=pbc_bonds) if (need_bonds and n > 1) \
        else (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3), int))
    bi, bj, shifts = bond_list
    T = shifts @ s.cell if (s.cell is not None and len(shifts)) else np.zeros((len(bi), 3))

    # ---- polyhedra (before bonds so their centre-ligand bonds can be hidden)
    hidden_pairs = set()
    if polyhedra:
        spec = {} if polyhedra is True else dict(polyhedra)
        centres = spec.get("centers", spec.get("centres"))
        if centres is None:
            centres = sorted({x for x in s.symbols if x not in NONMETALS})
        centres = set(np.atleast_1d(centres))
        ligands = spec.get("ligands")
        ligands = None if ligands is None else set(np.atleast_1d(ligands))
        # default ligands of a metal centre: its non-metal neighbours (the anions)
        alpha = spec.get("alpha", 0.38)
        min_lig = spec.get("min_ligands", 4)
        neigh = [[] for _ in range(n)]
        for i, j, tvec in zip(bi, bj, T):
            neigh[i].append((j, s.positions[j] + tvec))
            neigh[j].append((i, s.positions[i] - tvec))
        for c in range(n):
            if s.symbols[c] not in centres and s.ids[c] not in centres:
                continue
            if ligands is not None:
                ligs = [(r_, p) for r_, p in neigh[c] if s.symbols[r_] in ligands]
            elif s.symbols[c] not in NONMETALS:
                ligs = [(r_, p) for r_, p in neigh[c] if s.symbols[r_] in NONMETALS]
            else:
                ligs = list(neigh[c])
            if len(ligs) < min_lig:
                continue
            pts = np.array([p for _, p in ligs])
            colour = np.array(to_rgb(spec["color"])) if spec.get("color") else base[c]
            edge_c = edge_of(colour, spec.get("edgecolor", "darker"), st.darken)
            for face_idx in _hull_faces(pts):
                fxy, fd, _ = scene.project(pts[face_idx])
                t = float(scene.t(fd.mean()))
                fade = float(scene.fade(t, extra_fade[c]))
                scene.add(fd.mean(), -1, 0, _poly(fxy), scene.rgba(colour, fade, alpha),
                          scene.rgba(edge_c, fade, min(1.0, alpha + 0.45)), spec.get("edgewidth", 0.5),
                          gid=f"polyhedron-{s.ids[c]}")
            if spec.get("hide_bonds", True):
                hidden_pairs.update(frozenset((c, r_)) for r_, _ in ligs)

    # ---- bonds
    if bonds and mode != "spacefill" and len(bi):
        orders = _resolve_orders(s, bond_orders, bond_list)
        styles_by_ids = {}
        for key, value in (bond_styles or {}).items():
            value = _NAMED_BOND_STYLES[value] if isinstance(value, str) else dict(value)
            styles_by_ids[frozenset(key)] = value
        split = isinstance(st.bond_color, str) and st.bond_color == "split"
        bond_rgb = None if split else np.array(to_rgb(st.bond_color))
        for i, j, tvec, order in zip(bi, bj, T, orders):
            if frozenset((i, j)) in hidden_pairs:
                continue
            bst = styles_by_ids.get(frozenset((int(s.ids[i]), int(s.ids[j]))), {})
            if bst.get("hidden"):
                continue
            dashed_all = bool(bst.get("dashed"))
            if dashed_all:
                order = 1
            w = st.bond_width * bst.get("width", 1.0)
            own_rgb = np.array(to_rgb(bst["color"])) if bst.get("color") else None
            pi, pj = s.positions[i], s.positions[j] + tvec
            v = pj - pi
            L = np.linalg.norm(v)
            u = v / L
            pxy, _, _ = scene.project(np.array([pi, pj]))
            d2 = pxy[1] - pxy[0]
            n2 = np.hypot(*d2)
            perp = np.array([-d2[1], d2[0]]) / n2 if n2 > 1e-9 else np.zeros(2)
            if order > 1 and n2 < 0.25 * L:
                order = 1   # nearly along the line of sight: side-by-side lines would overlap
            for off, width, dashed in _MULTI.get(order, _MULTI[min(int(round(order)), 3)]):
                off, width = off * w, width * w
                dashed = dashed or dashed_all
                reach = abs(off) + width / 2
                si = np.sqrt(max(r_eff[i] ** 2 - reach ** 2, 0.0))
                sj = np.sqrt(max(r_eff[j] ** 2 - reach ** 2, 0.0))
                if L <= si + sj:
                    continue  # spheres overlap: bond hidden
                ai, aj = pi + u * si, pj - u * sj
                mid = 0.5 * (ai + aj)
                if split or tvec.any():
                    halves = [(ai, mid, i), (aj - tvec, mid - tvec, j)]
                else:
                    halves = [(ai, aj, None)]
                for start, end, owner in halves:
                    pieces = _dashes(start, end, st.dash_length, st.dash_gap) if dashed else [(start, end)]
                    for p0, p1 in pieces:
                        q, d, f = scene.project(np.array([p0, p1]))
                        dmid = float(d.mean())
                        if owner is None:
                            t = 1.0 if (hl[i] and hl[j]) else float(scene.t(dmid))
                            xf = max(extra_fade[i], extra_fade[j])
                        else:
                            t = 1.0 if hl[owner] else float(scene.t(dmid))
                            xf = extra_fade[owner]
                        k = float(scene.k(t)) * float(f.mean())
                        shift2 = perp * off * k
                        poly = _bond_polygon(q[0] + shift2, q[1] + shift2, 0.5 * width * k)
                        if poly is None:
                            continue
                        if own_rgb is not None:
                            face = own_rgb
                        elif split and owner is not None:
                            face = base[owner]
                        else:
                            face = bond_rgb if bond_rgb is not None else base[i]
                        edge = edge_of(face, st.bond_edgecolor, st.darken)
                        fade = float(scene.fade(t, xf))
                        scene.add(dmid, -1, 0, _poly(poly), scene.rgba(face, fade),
                                  scene.rgba(edge, fade), st.bond_edgewidth * k,
                                  gid=f"bond-{s.ids[i]}-{s.ids[j]}")

    # ---- dashed contacts (hydrogen bonds or given pairs)
    if contacts is not None:
        if isinstance(contacts, str):
            if contacts != "hbonds":
                raise ValueError("contacts must be 'hbonds' or a list of id pairs")
            pairs = find_hbonds(s, bond_list)
        else:
            pairs = [(s.index_of(a)[0], s.index_of(b)[0]) for a, b in contacts]
        crgb = np.array(to_rgb(st.contact_color))
        for i, j in pairs:
            pi, pj = s.positions[i], s.positions[j]
            v = pj - pi
            L = np.linalg.norm(v)
            if L <= r_eff[i] + r_eff[j]:
                continue
            u = v / L
            for p0, p1 in _dashes(pi + u * r_eff[i], pj - u * r_eff[j], st.dash_length, st.dash_gap):
                q, d, f = scene.project(np.array([p0, p1]))
                t = float(scene.t(d.mean()))
                k = float(scene.k(t)) * float(f.mean())
                poly = _bond_polygon(q[0], q[1], 0.5 * st.contact_width * k)
                if poly is not None:
                    scene.add(d.mean(), -1, 0, _poly(poly), scene.rgba(crgb, float(scene.fade(t))),
                              (0, 0, 0, 0), 0.0, gid=f"contact-{s.ids[i]}-{s.ids[j]}")

    # ---- rings
    if rings:
        spec = {} if rings is True else dict(rings)
        ring_rgb = np.array(to_rgb(spec.get("color", "#8c8c8c")))
        alpha = spec.get("alpha", 0.28)
        aromatic = spec.get("aromatic", False)
        for ring in find_rings(s, bond_list, max_size=spec.get("max_size", 8)):
            ring = np.asarray(ring)
            rxy, rd, _ = scene.project(s.positions[ring])
            dm = float(rd.mean())
            t = float(scene.t(dm))
            fade = float(scene.fade(t))
            scene.add(dm, -1, 0, _poly(rxy), scene.rgba(ring_rgb, fade, alpha),
                      (0, 0, 0, 0), 0.0, gid="ring")
            lengths = np.linalg.norm(s.positions[ring] - s.positions[np.roll(ring, 1)], axis=1)
            is_arom = aromatic is True or (aromatic == "auto" and len(ring) in (5, 6)
                                           and np.ptp(lengths) < 0.05)
            if is_arom:
                c = rxy.mean(axis=0)
                rad = 0.55 * np.linalg.norm(rxy - c, axis=1).mean()
                scene.add(dm + 1e-3, -1, 1, _circle(tuple(c), rad), (0, 0, 0, 0),
                          scene.rgba(edge_of(ring_rgb, "darker", 0.35), fade), 1.0, gid="ring")

    # ---- vectors
    if vectors is not None:
        vec = np.asarray(s.arrays[vectors] if isinstance(vectors, str) else vectors, float)
        if vec.shape != (n, 3):
            raise ValueError(f"vectors must be (N, 3) for {n} atoms, got {vec.shape}")
        for a in range(n):
            vv = vec[a] * vector_scale
            if not np.any(vv):
                continue
            colour = base[a] if (isinstance(vector_color, str) and vector_color == "atom") \
                else np.array(to_rgb(vector_color))
            _arrow(scene, s.positions[a], s.positions[a] + vv, vector_width, colour, fade_atom[a],
                   a, 60, gid=f"vector-{s.ids[a]}")

    # ---- vacancies
    if vacancies is not None:
        vac = np.atleast_2d(np.asarray(vacancies, float))
        vxy, vd, vf = scene.project(vac)
        for p, d, f in zip(vxy, vd, vf):
            t = float(scene.t(d))
            rad = st.vacancy_radius * float(scene.k(t)) * f
            # markers are annotations: always on top, or a vacancy under an atom would vanish
            scene.add(d + 1e6, -1, 5, _circle(tuple(p), rad), (0, 0, 0, 0),
                      scene.rgba((0.3, 0.3, 0.3), float(scene.fade(t))), 0.9, ls=DASHED, gid="vacancy")
            scene.extent.extend([p - rad, p + rad])

    # ---- cell, lattice vectors, planes
    extras = []
    if (cell or cell_vectors or planes) and s.cell is None:
        raise ValueError("cell, cell_vectors and planes need a structure with a unit cell")
    if cell:
        cr = np.array(to_rgb(st.cell_color))
        nseg = 12
        for p0, vec in _cell_edges(s.origin, *s.cell):
            for q_ in range(nseg):
                pq, dq, fq = scene.project(np.array([p0 + vec * q_ / nseg, p0 + vec * (q_ + 1) / nseg]))
                dmid = float(dq.mean())
                t = float(scene.t(dmid))
                k = float(scene.k(t)) * float(fq.mean())
                ext = (pq[1] - pq[0]) * 0.02
                poly = _bond_polygon(pq[0] - ext, pq[1] + ext, 0.5 * st.cell_width * k)
                if poly is not None:
                    scene.add(dmid, -2, 0, _poly(poly), scene.rgba(cr, float(scene.fade(t))),
                              (0, 0, 0, 0), 0.0, gid="cell")
            scene.extent.extend(scene.project(np.array([p0, p0 + vec]))[0])
    if cell_vectors:
        frac = 1.0 if cell_vectors is True else float(cell_vectors)
        for vec, name, colour in zip(s.cell, "abc", ("#c0392b", "#27ae60", "#2471a3")):
            tip = s.origin + vec * frac
            _arrow(scene, s.origin, tip, 1.4 * st.cell_width, np.array(to_rgb(colour)), 0.0, -3, 0, gid=f"cell-{name}")
            (txy,), _, _ = scene.project(tip[None, :] + 0.08 * vec[None, :] / max(np.linalg.norm(vec), 1e-9) * 3)
            extras.append(ax.text(*txy, name, color=colour, fontsize=9, fontstyle="italic",
                                  ha="center", va="center", zorder=zorder + 2))
    for spec in (planes or []):
        spec = {"hkl": spec} if not isinstance(spec, dict) else dict(spec)
        poly3 = _plane_polygon(s.cell, s.origin, spec["hkl"], spec.get("offset", 1.0))
        if poly3 is None:
            continue
        pxy, pd, _ = scene.project(poly3)
        colour = np.array(to_rgb(spec.get("color", "#8e44ad")))
        alpha = spec.get("alpha", 0.3)
        t = float(scene.t(pd.mean()))
        scene.add(pd.mean(), -1, 0, _poly(pxy), scene.rgba(colour, float(scene.fade(t)), alpha),
                  (*colour, min(1.0, alpha + 0.4)), 0.8, gid=f"plane-{''.join(map(str, spec['hkl']))}")
        scene.extent.extend(pxy)

    # ---- assemble, back to front
    items = scene.items
    if rasterized == "auto":
        rasterized = len(items) > AUTO_RASTERIZE
    coll = None
    art_patches = []
    if items:
        d = np.array([it[0] for it in items])
        g = np.array([it[1] for it in items])
        o = np.array([it[2] for it in items])
        order = np.lexsort((o, g, np.round(d, 6)))
        if editable:
            for rank, q_ in enumerate(order):
                _, _, _, path, face, edge, lw, ls, gid = items[q_]
                patch = PathPatch(path)
                patch.set_facecolor(face)
                patch.set_edgecolor(edge)
                patch.set_linewidth(lw)
                patch.set_linestyle(ls)
                patch.set_zorder(zorder + 0.999 * rank / len(order))
                patch.set_gid(gid)
                patch.set_rasterized(rasterized)
                ax.add_patch(patch)
                art_patches.append(patch)
        else:
            coll = PathCollection([items[q_][3] for q_ in order], zorder=zorder)
            coll.set_facecolor([items[q_][4] for q_ in order])
            coll.set_edgecolor([items[q_][5] for q_ in order])
            coll.set_linewidth([items[q_][6] for q_ in order])
            styles = [items[q_][7] for q_ in order]
            if any(ls != SOLID for ls in styles):
                coll.set_linestyle(styles)
            coll.set_joinstyle("round")
            coll.set_rasterized(rasterized)
            coll.set_gid("inkatoms-structure")
            ax.add_collection(coll)
    else:
        coll = PatchCollection([], zorder=zorder)
        ax.add_collection(coll)

    P = np.column_stack([xy, depth]) if n else np.zeros((0, 3))
    art = StructureArtist(ax, coll, s, P, rr, R, st, scene)
    art.patches = art_patches
    art.mappable = mappable
    art.values = values
    art.extras = extras

    if autoscale and n:
        x0, x1, y0, y1 = art.bounds
        if scene.extent:
            c = np.array(scene.extent)
            x0, x1 = min(x0, c[:, 0].min()), max(x1, c[:, 0].max())
            y0, y1 = min(y0, c[:, 1].min()), max(y1, c[:, 1].max())
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    if axis_off:
        ax.set_axis_off()
    if scale is not None:
        set_scale(ax, scale)
    if labels is not None:
        from .annotate import _draw_labels
        art.labels = _draw_labels(art, labels, values, base, fade_atom, dict(label_kw or {}), zorder)
    return art


def set_scale(ax, inches_per_angstrom, centre=None):
    """Set the axes limits so 1 Å is exactly inches_per_angstrom inches on the page,
    centred on the current view (or on centre=(x, y) in Å). Use the same value
    in every figure to keep structures at one scale across a paper."""
    fig = ax.figure
    box = ax.get_position(original=True)
    w_in = box.width * fig.get_figwidth()
    h_in = box.height * fig.get_figheight()
    if centre is None:
        centre = (np.mean(ax.get_xlim()), np.mean(ax.get_ylim()))
    hx = 0.5 * w_in / inches_per_angstrom
    hy = 0.5 * h_in / inches_per_angstrom
    ax.set_xlim(centre[0] - hx, centre[0] + hx)
    ax.set_ylim(centre[1] - hy, centre[1] + hy)
    return ax


def structure_figure(atoms, inches_per_angstrom=0.3, *, pad=0.4, max_size=12.0, **draw_kwargs):
    """A new figure sized to the structure at a fixed scale (inches per Å),
    with the structure filling it. Returns (fig, ax, artist)."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    art = draw(ax, atoms, pad=0.0, **draw_kwargs)
    x0, x1, y0, y1 = art.bounds
    w = (x1 - x0 + 2 * pad) * inches_per_angstrom
    h = (y1 - y0 + 2 * pad) * inches_per_angstrom
    if max(w, h) > max_size:
        plt.close(fig)
        raise ValueError(f"figure would be {w:.1f} x {h:.1f} in; lower inches_per_angstrom "
                         f"or raise max_size")
    fig.set_size_inches(w, h)
    set_scale(ax, inches_per_angstrom, centre=(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
    return fig, ax, art
