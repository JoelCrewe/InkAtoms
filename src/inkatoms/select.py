"""Cropping, centring and tiling. Atom ids are kept."""

from __future__ import annotations

import itertools

import numpy as np

from .bonds import coordination
from .structure import Structure, as_structure
from .view import _parse_axis, _unit


def _direction(axis):
    return _parse_axis(axis) if isinstance(axis, str) else _unit(axis)


def _point(s, at):
    """A point given as a 3-vector, or the centroid of the atoms with these ids."""
    if at is None:
        return s.positions.mean(axis=0)
    arr = np.asarray(at)
    if arr.shape == (3,) and arr.dtype.kind == "f":
        return arr.astype(float)
    rows = s.index_of(arr)
    if len(rows) == 0:
        raise ValueError(f"no atoms with ids {at}")
    return s.positions[rows].mean(axis=0)


def _crop(s, mask, keep):
    if keep is not None:
        mask = mask | np.isin(s.ids, np.atleast_1d(keep))
    return s[mask].copy(pbc=False)


def slab(atoms, axis="y", half_width=4.0, *, at=None, keep=None):
    """Keep a slab of atoms within half_width (Å) of a plane.

    axis: plane normal, e.g. "y" or a vector (usually the view direction).
    at:   point the plane passes through (3-vector or atom ids); default the centroid.
    keep: ids to keep whatever their position (e.g. the atoms you highlight).
    """
    s = as_structure(atoms)
    n = _direction(axis)
    d = (s.positions - _point(s, at)) @ n
    return _crop(s, np.abs(d) <= half_width, keep)


def around(atoms, at, radius=6.0, *, keep=None):
    """Keep atoms within radius (Å) of a point or of the centroid of some atom ids."""
    s = as_structure(atoms)
    d = np.linalg.norm(s.positions - _point(s, at), axis=1)
    return _crop(s, d <= radius, keep)


def box(atoms, lo, hi, *, keep=None):
    """Keep atoms inside an axis-aligned box lo <= r <= hi (use +-inf to leave an axis open)."""
    s = as_structure(atoms)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    mask = np.all((s.positions >= lo) & (s.positions <= hi), axis=1)
    return _crop(s, mask, keep)


def centre(atoms, at=None):
    """Translate so that a point (or the centroid of some atom ids) is at the origin."""
    s = as_structure(atoms)
    return s.copy(positions=s.positions - _point(s, at))


center = centre


def wrap(atoms):
    """Wrap atoms into the unit cell along periodic directions."""
    s = as_structure(atoms)
    if s.cell is None:
        return s
    frac = s.positions @ np.linalg.inv(s.cell)
    frac[:, s.pbc] %= 1.0
    return s.copy(positions=frac @ s.cell)


def tile(atoms, reps=(3, 3, 1), *, centred=True):
    """Repeat a periodic cell reps[0] x reps[1] x reps[2] times.

    The copy in the original cell keeps its ids, so highlights and xy(id)
    still pick out that one atom; copy k gets id + k * stride (stride =
    max id + 1). With centred=True the original cell sits in the middle of the
    tiled block, so a crop around it has whole bonds.
    """
    s = as_structure(atoms)
    if s.cell is None:
        raise ValueError("tile needs a structure with a unit cell")
    reps = np.asarray(reps, int)
    offsets = np.array(list(itertools.product(*(range(r) for r in reps))))
    if centred:
        offsets = offsets - (reps - 1) // 2
    shifts = offsets @ s.cell
    pos = (s.positions[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    k = len(offsets)
    cell = s.cell * reps[:, None]
    origin = s.origin + (offsets.min(axis=0) @ s.cell)
    stride = int(s.ids.max()) + 1 if len(s) else 0
    home = int(np.flatnonzero(~offsets.any(axis=1))[0])   # the (0, 0, 0) copy
    copy_no = np.empty(k, int)
    copy_no[[home] + [q for q in range(k) if q != home]] = np.arange(k)
    ids = (s.ids[None, :] + copy_no[:, None] * stride).reshape(-1)
    arrays = {key: np.concatenate([v] * k) for key, v in s.arrays.items()}
    occ = None if s.occupancy is None else tuple(s.occupancy) * k
    return Structure(np.tile(s.symbols, k), pos, cell=cell, pbc=s.pbc, ids=ids, origin=origin,
                     arrays=arrays, occupancy=occ, bond_orders=s.bond_orders)


def images(atoms, width=1.5, *, tol=1e-3):
    """Add the periodic images that lie within width (Å) outside the cell.

    The added atoms are marked in arrays["_image"] (draw() fades them as
    ghosts) and get new ids (id + k * stride). width=0 gives VESTA-style
    boundary atoms: only the copies of atoms sitting on cell faces, edges
    and corners.
    """
    s = as_structure(atoms)
    if s.cell is None:
        raise ValueError("images needs a structure with a unit cell")
    inv = np.linalg.inv(s.cell)
    frac = (s.positions - s.origin) @ inv
    # fractional width along each axis
    vol = abs(np.linalg.det(s.cell))
    heights = np.array([vol / np.linalg.norm(np.cross(s.cell[(k + 1) % 3], s.cell[(k + 2) % 3]))
                        for k in range(3)])
    fw = (width + tol) / heights
    reach = [range(-1, 2) if s.pbc[k] else range(0, 1) for k in range(3)]
    stride = int(s.ids.max()) + 1 if len(s) else 0
    parts, copy_no = [], 1
    for shift in itertools.product(*reach):
        shift = np.array(shift)
        if not shift.any():
            continue
        f = frac + shift
        inside = np.all((f >= -fw) & (f <= 1 + fw), axis=1)
        if inside.any():
            sub = s[inside]
            parts.append(sub.copy(positions=sub.positions + shift @ s.cell,
                                  ids=sub.ids + copy_no * stride))
        copy_no += 1
    base = s.copy(arrays={**s.arrays, "_image": np.zeros(len(s), bool)})
    for part in parts:
        base = base.concat(part.copy(arrays={**part.arrays, "_image": np.ones(len(part), bool)}))
    return base.copy(pbc=False)


def boundary(atoms, tol=1e-3):
    """VESTA-style boundary atoms: repeat atoms on cell faces, edges and corners."""
    return images(atoms, width=0.0, tol=tol)


def make_whole(atoms, **bond_kwargs):
    """Unwrap molecules split by the periodic boundary so each bonded cluster is
    drawn in one piece (cluster centres stay inside the cell where possible)."""
    s = as_structure(atoms)
    if s.cell is None or not s.pbc.any():
        return s
    from .bonds import find_bonds

    i, j, shift = find_bonds(s, **bond_kwargs)
    nbrs = [[] for _ in range(len(s))]
    for a, b, t in zip(i, j, shift):
        nbrs[a].append((b, t))
        nbrs[b].append((a, -t))
    image = np.zeros((len(s), 3), int)
    seen = np.zeros(len(s), bool)
    for start in range(len(s)):
        if seen[start]:
            continue
        seen[start] = True
        stack, members = [start], [start]
        while stack:
            a = stack.pop()
            for b, t in nbrs[a]:
                if not seen[b]:
                    seen[b] = True
                    image[b] = image[a] + t
                    stack.append(b)
                    members.append(b)
        # put the cluster centre back in the home cell
        members = np.array(members)
        centre = (s.positions[members] + image[members] @ s.cell).mean(axis=0)
        cfrac = (centre - s.origin) @ np.linalg.inv(s.cell)
        image[members] -= np.floor(cfrac).astype(int) * s.pbc
    return s.copy(positions=s.positions + image @ s.cell)


def drop_unbonded(atoms, min_bonds=1, **bond_kwargs):
    """Remove atoms with fewer than min_bonds bonds (e.g. strays left by a crop)."""
    s = as_structure(atoms)
    return s[coordination(s, **bond_kwargs) >= min_bonds]
