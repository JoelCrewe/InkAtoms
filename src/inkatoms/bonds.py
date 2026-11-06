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


def coordination(atoms, **kwargs):
    """Number of bonds on each atom."""
    s = as_structure(atoms)
    i, j, _ = find_bonds(s, **kwargs)
    return np.bincount(np.concatenate([i, j]), minlength=len(s))
