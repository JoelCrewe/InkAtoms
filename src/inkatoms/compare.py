"""Compare two structures: match their atoms, align them, and draw the change
(structure B, with A as faint ghosts and arrows from A to B)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .bonds import find_bonds
from .structure import Structure, as_structure


def kabsch(P, Q, weights=None):
    """Rotation R and translation t minimising |P @ R.T + t - Q| (rows matched)."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    cp, cq = w @ P, w @ Q
    H = (P - cp).T @ ((Q - cq) * w[:, None])
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - cp @ R.T


def _assign(a, b, cost_fn):
    """Optimal one-to-one matching of rows, element by element."""
    ra, rb = [], []
    for sym in sorted(set(a.symbols) & set(b.symbols)):
        ia, ib = np.flatnonzero(a.symbols == sym), np.flatnonzero(b.symbols == sym)
        r, c = linear_sum_assignment(cost_fn(ia, ib))
        ra.extend(ia[r])
        rb.extend(ib[c])
    order = np.argsort(ra)
    return np.asarray(ra, int)[order], np.asarray(rb, int)[order]


def match(a, b, method="auto", *, iterations=4, descriptor="fingerprint", env_weight=1.0, known=None,
          pairs=None):
    """Pair the atoms of a with those of b; returns (rows_a, rows_b).

    method  "ids"       atoms with the same id (persistent ids, e.g. "uid")
            "distance"  optimal assignment on distance per element, refined by
                        aligning and re-matching (for structures without shared ids)
            "environment"  optimal assignment on local-environment descriptors only
                        (inkatoms.ml), for structures that are not in the same place;
                        fails on symmetric structures where many atoms look alike
            "hybrid"    distance and environment together (env_weight sets the
                        balance): the most robust when atoms moved by about a bond length
            "auto"      ids if most ids are shared, else distance
    known   {id_a: id_b} pairs you are sure of; they are fixed and the method
            matches only the remaining atoms
    pairs   a complete mapping [(id_a, id_b), ...] given by you (no matching done)
    """
    a, b = as_structure(a), as_structure(b)
    if pairs is not None:
        known, method = dict(pairs), "known-only"
    if known:
        row_a = {int(i): r for r, i in enumerate(a.ids)}
        row_b = {int(i): r for r, i in enumerate(b.ids)}
        missing = [k for k, v in known.items() if int(k) not in row_a or int(v) not in row_b]
        if missing:
            raise KeyError(f"known pairs refer to ids not present: {missing[:6]}")
        fa = np.array([row_a[int(k)] for k in known], int)
        fb = np.array([row_b[int(v)] for v in known.values()], int)
        if method == "known-only":
            order = np.argsort(fa)
            return fa[order], fb[order]
        rest_a = np.setdiff1d(np.arange(len(a)), fa)
        rest_b = np.setdiff1d(np.arange(len(b)), fb)
        ra, rb = match(a[rest_a], b[rest_b], method=method, iterations=iterations, descriptor=descriptor,
                       env_weight=env_weight)
        ra, rb = np.concatenate([fa, rest_a[ra]]), np.concatenate([fb, rest_b[rb]])
        order = np.argsort(ra)
        return ra[order], rb[order]
    if method == "auto":
        shared = len(set(map(int, a.ids)) & set(map(int, b.ids)))
        method = "ids" if shared >= 0.9 * min(len(a), len(b)) else "distance"
    if method == "ids":
        row_b = {int(i): r for r, i in enumerate(b.ids)}
        ra = [r for r, i in enumerate(a.ids) if int(i) in row_b and a.symbols[r] == b.symbols[row_b[int(i)]]]
        return np.array(ra, int), np.array([row_b[int(a.ids[r])] for r in ra], int)
    if method == "distance":
        pa = a.positions - a.positions.mean(axis=0)
        pb = b.positions - b.positions.mean(axis=0)
        for _ in range(iterations):
            ra, rb = _assign(a, b, lambda ia, ib: cdist(pa[ia], pb[ib]))
            R, t = kabsch(pa[ra], pb[rb])
            pa = pa @ R.T + t
        return ra, rb
    if method in ("environment", "hybrid"):
        from .ml import descriptors

        species = sorted(set(a.symbols) | set(b.symbols))
        da = descriptors(a, method=descriptor, species=species)
        db = descriptors(b, method=descriptor, species=species)
        sd = np.vstack([da, db]).std(axis=0) + 1e-9
        da, db = da / sd, db / sd
        if method == "environment":
            return _assign(a, b, lambda ia, ib: cdist(da[ia], db[ib]))
        # hybrid: align by distance first, then match on position and environment together
        ra, rb = match(a, b, "distance", iterations=iterations)
        R, t = kabsch(a.positions[ra], b.positions[rb])
        pa = a.positions @ R.T + t

        def cost(ia, ib):
            dd = cdist(pa[ia], b.positions[ib])
            de = cdist(da[ia], db[ib])
            return dd / (np.median(dd) + 1e-9) + env_weight * de / (np.median(de) + 1e-9)
        return _assign(a, b, cost)
    raise ValueError("method must be 'auto', 'ids', 'distance' or 'environment'")


@dataclass
class Comparison:
    """The result of compare(): how B differs from A."""

    a: Structure
    b: Structure
    rows_a: np.ndarray          # matched rows of a ...
    rows_b: np.ndarray          # ... and the rows of b they correspond to
    rotation: np.ndarray        # applied to a to align it with b
    translation: np.ndarray
    method: str

    @property
    def a_aligned(self):
        """a's positions moved onto b's frame."""
        return self.a.positions @ self.rotation.T + self.translation

    @property
    def displacement(self):
        """(n_matched, 3) vectors from each atom in (aligned) a to its partner in b."""
        return self.b.positions[self.rows_b] - self.a_aligned[self.rows_a]

    @property
    def shift(self):
        return np.linalg.norm(self.displacement, axis=1)

    @property
    def rmsd(self):
        return float(np.sqrt(np.mean(self.shift ** 2))) if len(self.rows_a) else float("nan")

    def largest(self, n=5):
        """(id in b, symbol, shift in Å) of the atoms that moved most."""
        order = np.argsort(self.shift)[::-1][:n]
        return [(int(self.b.ids[self.rows_b[k]]), str(self.b.symbols[self.rows_b[k]]), float(self.shift[k]))
                for k in order]

    def __str__(self):
        lines = [f"{len(self.rows_a)} atoms matched ({self.method}), RMSD {self.rmsd:.3f} Å after alignment",
                 f"{len(self.a) - len(self.rows_a)} unmatched in A, {len(self.b) - len(self.rows_b)} in B"]
        lines += [f"  {sym:2s} {i:6d}  moved {d:.2f} Å" for i, sym, d in self.largest(5)]
        return "\n".join(lines)


def compare(a, b, method="auto", *, align=True, align_on=None, ids_a=None, ids_b=None, **match_kwargs):
    """Match a's atoms to b's, align a onto b (Kabsch), and return a Comparison.

    align_on: ids (in a) to fit the alignment on, e.g. a rigid substrate, so the
    displacement shows how the rest moved relative to it.
    ids_a, ids_b: which ids identify atoms in each structure: an array, or the name of
    a per-atom array (e.g. "uid") for ASE input. known= and pairs= go to match().
    """
    a, b = as_structure(a, ids=ids_a), as_structure(b, ids=ids_b)
    used = method if method != "auto" else None
    if match_kwargs.get("pairs") is not None:
        used = "given pairs"
    ra, rb = match(a, b, method=method, **match_kwargs)
    if used is None:
        shared = len(set(map(int, a.ids)) & set(map(int, b.ids)))
        used = "ids" if shared >= 0.9 * min(len(a), len(b)) else "distance"
    if align and len(ra) >= 3:
        sel = np.ones(len(ra), bool) if align_on is None else np.isin(a.ids[ra], np.atleast_1d(align_on))
        R, t = kabsch(a.positions[ra[sel]], b.positions[rb[sel]])
    else:
        R, t = np.eye(3), np.zeros(3)
    return Comparison(a, b, ra, rb, R, t, used)


def draw_comparison(ax, a, b, *, comparison=None, method="auto", align=True, align_on=None, ids_a=None,
                    ids_b=None, known=None, pairs=None, ghost=0.65,
                    ghost_scale=0.8,
                    arrows=True, arrow_scale=1.0, arrow_max=None, min_shift=0.15, color_by="shift",
                    cmap="magma_r",
                    vmin=0.0, vmax=None, arrow_color="#333333", cutoffs=None, bond_scale=1.15, **draw_kwargs):
    """Draw b with a as faint ghosts and arrows showing how each atom moved.

    ghost      how far a's atoms fade (0 = as strong as b, 1 = invisible); they are
               drawn behind b and ghost_scale times smaller
    min_shift  shifts below this (Å) get no arrow
    arrow_scale, arrow_max  arrows are arrow_scale x the shift, capped at arrow_max Å
               (so one atom that moved far does not swamp small relaxations)
    color_by   "shift" colours b's atoms by how far they moved (with a colorbar via
               inkatoms.colorbar), None keeps element colours
    Other keyword arguments go to draw(). Returns (artist, comparison).
    """
    from .render import draw

    a, b = as_structure(a, ids=ids_a), as_structure(b, ids=ids_b)
    cmp_ = comparison or compare(a, b, method=method, align=align, align_on=align_on, known=known, pairs=pairs)
    stride = int(max(b.ids.max(), a.ids.max())) + 1
    a_al = a.copy(positions=cmp_.a_aligned, ids=a.ids + stride, arrays={}, occupancy=None, bond_orders=None)
    b_clean = b.copy(arrays={}, occupancy=None)
    both = b_clean.concat(a_al).copy(cell=None, pbc=False)
    nb = len(b)
    fade = np.zeros(len(both))
    fade[nb:] = ghost
    layer = np.zeros(len(both))
    layer[nb:] = -1.0                    # ghosts behind the new structure ...
    scale = np.ones(len(both))
    scale[nb:] = ghost_scale             # ... and a little smaller, so they never hide it
    shift = np.full(len(both), np.nan)
    shift[cmp_.rows_b] = cmp_.shift
    vec = np.zeros((len(both), 3))
    if arrows:
        keep = cmp_.shift >= min_shift
        arrows_v = cmp_.displacement[keep] * arrow_scale
        if arrow_max is not None:
            length = np.linalg.norm(arrows_v, axis=1, keepdims=True)
            arrows_v = arrows_v * np.minimum(1.0, arrow_max / np.maximum(length, 1e-12))
        vec[nb + cmp_.rows_a[keep]] = arrows_v
    both = both.copy(arrays={"_fade": fade, "_layer": layer, "_scale": scale, "shift": shift})
    pairs = []
    for part in (b, a_al):
        i, j, _ = find_bonds(part, cutoffs=cutoffs, scale=bond_scale, pbc=False)
        pairs.extend(zip(part.ids[i], part.ids[j]))
    kw = dict(draw_kwargs)
    if color_by == "shift":
        kw.update(color_by="shift", cmap=cmap, vmin=vmin, vmax=vmax)
    elif color_by is not None:
        kw["color_by"] = color_by
    art = draw(ax, both, bonds=pairs, vectors=vec if arrows else None, vector_color=arrow_color,
               vectors_front=True, **kw)
    return art, cmp_
