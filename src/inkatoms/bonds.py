"""Bond detection with a KD-tree, aware of periodic boundaries."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree

from ._elements import DATA
from .structure import Structure, as_structure


def _pair_key(key):
    if isinstance(key, frozenset):
        return key          # frozenset({"Mo"}) is the Mo-Mo pair
    if isinstance(key, str):
        a, b = key.replace(" ", "").split("-")
    else:
        a, b = key
    return frozenset((a, b))


NONMETALS = frozenset("H He B C N O F Ne Si P S Cl Ar Ge As Se Br Kr Sb Te I Xe At Rn".split())


def cutoff_matrix(species, cutoffs=None, scale=1.15, metal_scale=None):
    """Cutoff (Å) for every pair of species.

    Default: scale x (covalent radius a + covalent radius b). Metal-metal
    pairs use metal_scale instead, because metal covalent radii are already
    generous: 1.0 in metals and alloys, and 0.92 when non-metals other than H
    are present (compounds), so cation-cation contacts in oxides and
    sulfides (rutile Ti-Ti 2.96 Å, MoS2 Mo-Mo 3.16 Å) are not bonds while
    real metal-metal bonds (Mo3 triangles, 2.7 Å) are. Entries in `cutoffs`
    override pairs, keyed "Mo-S", ("Mo", "S") or frozenset; 0 switches a pair off.
    """
    species = list(species)
    if metal_scale is None:
        compound = any(s in NONMETALS and s != "H" for s in species)
        metal_scale = 0.92 if compound else 1.0
    cov = np.array([DATA[s][0] for s in species])
    metal = np.array([s not in NONMETALS for s in species])
    factor = np.where(metal[:, None] & metal[None, :], metal_scale, scale)
    mat = factor * (cov[:, None] + cov[None, :])
    for key, value in (cutoffs or {}).items():
        pair = _pair_key(key)
        for i, a in enumerate(species):
            for j, b in enumerate(species):
                if frozenset((a, b)) == pair:
                    mat[i, j] = value
    return mat


def find_bonds(atoms, cutoffs=None, scale=1.15, pbc=None, metal_scale=None):
    """Find bonded pairs.

    Returns (i, j, shift): row indices into the structure and, for each bond,
    the integer lattice shift such that atom j's partner image sits at
    positions[j] + shift @ cell. Each bond appears once.

    pbc: None follows the structure; False ignores periodicity.
    """
    s = as_structure(atoms)
    n = len(s)
    empty = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3), int))
    if n < 1:
        return empty
    species, sp = np.unique(s.symbols, return_inverse=True)
    cut = cutoff_matrix(species, cutoffs, scale, metal_scale)
    rmax = float(cut.max())
    if rmax <= 0:
        return empty

    periodic = s.pbc.copy() if pbc is None else np.broadcast_to(np.asarray(pbc, bool), (3,)) & s.pbc
    if s.cell is None:
        periodic[:] = False

    reps = [0, 0, 0]
    if periodic.any():
        cell = s.cell
        vol = abs(np.linalg.det(cell))
        for k in range(3):
            if periodic[k]:
                width = vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3]))
                reps[k] = int(np.ceil(rmax / width))

    tree = cKDTree(s.positions)
    I, J, S = [], [], []
    for shift in itertools.product(*(range(-r, r + 1) for r in reps)):
        shift = np.array(shift)
        # keep one of each (i, j, +S) / (j, i, -S) pair
        nz = shift[np.nonzero(shift)[0]]
        if len(nz) and nz[0] < 0:
            continue
        other = s.positions + (shift @ s.cell if periodic.any() else 0.0)
        dist = tree.sparse_distance_matrix(cKDTree(other), rmax, output_type="ndarray")
        if len(dist) == 0:
            continue
        i, j, d = dist["i"], dist["j"], dist["v"]
        keep = (d > 1e-6) & (d <= cut[sp[i], sp[j]])
        if not shift.any():
            keep &= i < j
        I.append(i[keep])
        J.append(j[keep])
        S.append(np.repeat(shift[None, :], keep.sum(), axis=0))
    if not I:
        return empty
    return np.concatenate(I), np.concatenate(J), np.concatenate(S).astype(int)


# Longest distance (Å) still drawn as a double / triple bond, per element pair.
# Set between typical single, double and triple bond lengths and checked on
# ASE's g2 molecules: ethene, HCHO, CO2, CS2, DMSO, SO2 -> double; HCN, N2,
# nitriles -> triple; thiophene and pyrrole get alternating double bonds,
# while benzene and pyridine (1.39-1.40 Å, aromatic) stay single.
ORDER_LIMITS = {
    frozenset(("C",)): (1.385, 1.28),
    frozenset(("C", "N")): (1.31, 1.22),
    frozenset(("C", "O")): (1.29, 1.15),
    frozenset(("C", "S")): (1.64, None),
    frozenset(("N",)): (1.33, 1.19),
    frozenset(("N", "O")): (1.27, None),
    frozenset(("O",)): (1.35, None),
    frozenset(("S", "O")): (1.55, None),
    frozenset(("P", "O")): (1.53, None),
    frozenset(("S",)): (1.95, None),
}


def bond_orders(atoms, bonds=None, limits=None):
    """Guess bond orders (1, 2 or 3) from bond lengths.

    A length-based guess for drawing, not chemistry: it needs reasonable
    geometries (optimised or experimental, not hot MD frames) and only knows
    the pairs in ORDER_LIMITS (C, N, O, S, P); everything else is single.
    bonds: (i, j, shift) from find_bonds (found if not given).
    limits: {pair: (max double, max triple or None)} merged over ORDER_LIMITS.
    """
    s = as_structure(atoms)
    i, j, shift = find_bonds(s) if bonds is None else bonds
    table = dict(ORDER_LIMITS)
    for key, value in (limits or {}).items():
        table[_pair_key(key)] = value
    T = shift @ s.cell if (s.cell is not None and len(shift)) else 0.0
    d = np.linalg.norm(s.positions[j] + T - s.positions[i], axis=1)
    orders = np.ones(len(i), int)
    for k in range(len(i)):
        lim = table.get(frozenset((s.symbols[i[k]], s.symbols[j[k]])))
        if lim is None:
            continue
        double, triple = lim
        if triple is not None and d[k] <= triple:
            orders[k] = 3
        elif double is not None and d[k] <= double:
            orders[k] = 2
    return orders


def find_hbonds(atoms, bonds=None, *, max_ha=2.5, min_angle=120.0, donors=("N", "O", "F"),
                acceptors=("N", "O", "F")):
    """Hydrogen bonds D-H...A as (row of H, row of A) pairs.

    H must be covalently bonded to a donor; H...A within max_ha (Å) and the
    D-H...A angle at least min_angle (degrees). Periodic images are ignored.
    """
    s = as_structure(atoms)
    i, j, shift = find_bonds(s) if bonds is None else bonds
    sym = s.symbols
    acc_rows = np.flatnonzero(np.isin(sym, acceptors))
    if len(acc_rows) == 0:
        return []
    tree = cKDTree(s.positions[acc_rows])
    pairs = []
    for a, b, t in zip(i, j, shift):
        if t.any():
            continue
        for h, d in ((a, b), (b, a)):
            if sym[h] != "H" or sym[d] not in donors:
                continue
            for q in tree.query_ball_point(s.positions[h], max_ha):
                acc = acc_rows[q]
                if acc == d:
                    continue
                v1 = s.positions[d] - s.positions[h]
                v2 = s.positions[acc] - s.positions[h]
                dist = np.linalg.norm(v2)
                if dist < 1.3:
                    continue
                ang = np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * dist), -1, 1)))
                if ang >= min_angle:
                    pairs.append((int(h), int(acc)))
    return pairs


def find_rings(atoms, bonds=None, *, max_size=8, include_h=False):
    """Small rings as lists of row indices in ring order (the shortest ring
    through each bond, so fused rings come out separately)."""
    s = as_structure(atoms)
    i, j, shift = find_bonds(s) if bonds is None else bonds
    keep = ~shift.any(axis=1)
    if not include_h:
        keep &= (s.symbols[i] != "H") & (s.symbols[j] != "H")
    nbrs = {}
    for a, b in zip(i[keep], j[keep]):
        nbrs.setdefault(int(a), set()).add(int(b))
        nbrs.setdefault(int(b), set()).add(int(a))
    rings, seen = [], set()
    for a, b in zip(i[keep], j[keep]):
        a, b = int(a), int(b)
        # shortest path a -> b that does not use the a-b bond
        prev = {a: None}
        queue = [a]
        found = False
        while queue and not found:
            nxt = []
            for u in queue:
                for w in nbrs.get(u, ()):
                    if (u == a and w == b) or w in prev:
                        continue
                    prev[w] = u
                    if w == b:
                        found = True
                        break
                    nxt.append(w)
                if found:
                    break
            queue = nxt
            if len(prev) > 4000:
                break
        if not found:
            continue
        path = [b]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        if len(path) <= max_size and frozenset(path) not in seen:
            seen.add(frozenset(path))
            rings.append(path[::-1])
    return rings


def coordination(atoms, **kwargs):
    """Number of bonds on each atom."""
    s = as_structure(atoms)
    i, j, _ = find_bonds(s, **kwargs)
    return np.bincount(np.concatenate([i, j]), minlength=len(s))
