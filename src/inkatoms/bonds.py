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


def cutoff_matrix(species, cutoffs=None, scale=1.15, metal_scale=1.0):
    """Cutoff (Å) for every pair of species.

    Default: scale x (covalent radius a + covalent radius b), or metal_scale x
    the sum for metal-metal pairs (metal covalent radii are already generous,
    so e.g. the 3.16 Å Mo-Mo contacts in MoS2 are not bonds but the 2.7 Å
    Mo3 triangles are). Entries in `cutoffs` override pairs, keyed "Mo-S",
    ("Mo", "S") or frozenset; use 0 to switch a pair off.
    """
    species = list(species)
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


def find_bonds(atoms, cutoffs=None, scale=1.15, pbc=None, metal_scale=1.0):
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


def coordination(atoms, **kwargs):
    """Number of bonds on each atom."""
    s = as_structure(atoms)
    i, j, _ = find_bonds(s, **kwargs)
    return np.bincount(np.concatenate([i, j]), minlength=len(s))
