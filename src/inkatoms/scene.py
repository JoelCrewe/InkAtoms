"""The depth-sorted scene every drawing is built in, and the geometry helpers
the layers share (raw paths, bond polygons, dashes, arrows, hulls, planes)."""

from __future__ import annotations

import itertools

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.path import Path
from scipy.spatial import ConvexHull, cKDTree

from .bonds import coordination
from .bonds import bond_orders as guess_orders
from .styles import mix

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



# ---------------------------------------------------------------- scene

class Scene:
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
            if st.fade_distance:
                # a fixed fade length, so small and large structures fade alike
                self.far = self.near - float(st.fade_distance)
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

    atom_layer = None   # per-atom draw layer (set by draw when layers= is given)

    def add(self, depth, group, order, patch, face, edge, lw, ls=SOLID, gid=None, layer=None):
        if layer is None:
            layer = self.atom_layer[group] if (self.atom_layer is not None and group >= 0) else 0
        self.items.append((float(depth), group, order, patch, face, edge, lw, ls, gid, layer))


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


def _arrow(scene, p0, p1, width, colour, fade, group, order, depth=None, gid=None, layer=None):
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
                  face, (0, 0, 0, 0), 0.0, gid=gid, layer=layer)
    scene.add(dd, group, order + 1, _poly([b, neck + perp * head_w / 2, neck - perp * head_w / 2]),
              face, (0, 0, 0, 0), 0.0, gid=gid, layer=layer)
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
