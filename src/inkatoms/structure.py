"""The internal structure container and converters from ASE / pymatgen / arrays."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ._elements import DATA


@dataclass(frozen=True)
class Structure:
    """Atoms as plain arrays.

    ids are carried through every selection, so an atom keeps the same id
    after cropping or tiling and highlights can refer to the original indices.
    """

    symbols: np.ndarray            # (N,) str
    positions: np.ndarray          # (N, 3) float, Å
    cell: np.ndarray | None = None  # (3, 3) rows are lattice vectors, or None
    pbc: np.ndarray = None          # (3,) bool
    ids: np.ndarray = None          # (N,) int
    origin: np.ndarray = None       # (3,) corner of the cell, Å

    def __post_init__(self):
        sym = np.asarray(self.symbols, dtype=str)
        pos = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        if len(sym) != len(pos):
            raise ValueError(f"{len(sym)} symbols but {len(pos)} positions")
        unknown = sorted(set(sym) - set(DATA))
        if unknown:
            raise ValueError(f"unknown element symbols: {unknown}")
        cell = None if self.cell is None else np.asarray(self.cell, dtype=float).reshape(3, 3)
        if cell is not None and not np.any(cell):
            cell = None
        pbc = np.zeros(3, bool) if self.pbc is None else np.broadcast_to(np.asarray(self.pbc, bool), (3,)).copy()
        if cell is None:
            pbc[:] = False
        ids = np.arange(len(sym)) if self.ids is None else np.asarray(self.ids, dtype=int)
        if len(ids) != len(sym):
            raise ValueError(f"{len(ids)} ids for {len(sym)} atoms")
        object.__setattr__(self, "symbols", sym)
        object.__setattr__(self, "positions", pos)
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "pbc", pbc)
        object.__setattr__(self, "ids", ids)
        origin = np.zeros(3) if self.origin is None else np.asarray(self.origin, dtype=float).reshape(3)
        object.__setattr__(self, "origin", origin)

    def __len__(self):
        return len(self.symbols)

    def __getitem__(self, index):
        """Subset by mask or index array; keeps cell and pbc."""
        index = np.asarray(index)
        if index.dtype != bool:
            index = np.atleast_1d(index)
        return replace(self, symbols=self.symbols[index],
                       positions=self.positions[index], ids=self.ids[index])

    def copy(self, **changes):
        return replace(self, **changes)

    def index_of(self, ids):
        """Row indices of the atoms with these ids (every copy, if tiled)."""
        return np.flatnonzero(np.isin(self.ids, np.atleast_1d(ids)))


def as_structure(obj, *, ids=None) -> Structure:
    """Convert an ASE Atoms, pymatgen Structure/Molecule, (symbols, positions)
    pair, or Structure into a Structure.

    ids: an array of per-atom ids, or (for ASE) the name of a per-atom array
    such as "uid" to use as ids. Defaults to 0..N-1.
    """
    if isinstance(obj, Structure):
        return obj if ids is None else obj.copy(ids=ids)

    # ASE Atoms
    if hasattr(obj, "get_chemical_symbols") and hasattr(obj, "get_positions"):
        if isinstance(ids, str):
            ids = obj.arrays[ids]
        cell = np.asarray(obj.cell[:], dtype=float)
        return Structure(obj.get_chemical_symbols(), obj.get_positions(),
                         cell=cell, pbc=np.asarray(obj.pbc), ids=ids)

    # pymatgen Structure / Molecule
    if hasattr(obj, "cart_coords") and hasattr(obj, "sites"):
        symbols = [site.specie.symbol for site in obj.sites]
        lattice = getattr(obj, "lattice", None)
        cell = None if lattice is None else np.asarray(lattice.matrix)
        pbc = getattr(lattice, "pbc", (True, True, True)) if lattice is not None else False
        return Structure(symbols, obj.cart_coords, cell=cell, pbc=pbc, ids=ids)

    # (symbols, positions)
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        symbols, positions = obj
        return Structure(symbols, positions, ids=ids)

    raise TypeError(f"cannot make a Structure from {type(obj).__name__}; "
                    "pass ASE Atoms, a pymatgen Structure/Molecule or (symbols, positions)")
