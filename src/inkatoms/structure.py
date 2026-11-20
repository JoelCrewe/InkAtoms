"""Structure container and converters from ASE, pymatgen, RDKit and MDAnalysis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import numpy as np

from ._elements import DATA


@dataclass(frozen=True)
class Structure:
    """Atoms as plain arrays.

    Attributes
    ----------
    symbols, positions : (N,) str and (N, 3) float arrays (Å)
    cell : (3, 3) array of lattice vectors (rows), or None
    pbc : (3,) bool
    ids : (N,) int
        Kept through every selection, so highlights still refer to the original atoms.
    origin : (3,) float
        Corner of the cell.
    arrays : dict
        Extra per-atom data, e.g. ``charges``, ``forces`` or ``fixed``.
    occupancy : tuple or None
        Per site, None or ``((symbol, fraction), ...)`` for mixed sites.
    bond_orders : dict or None
        Known bond orders ``{(id_a, id_b): order}``, e.g. from RDKit.
    """

    symbols: np.ndarray
    positions: np.ndarray
    cell: np.ndarray | None = None
    pbc: np.ndarray = None
    ids: np.ndarray = None
    origin: np.ndarray = None
    arrays: dict = field(default_factory=dict)
    occupancy: tuple | None = None
    bond_orders: dict | None = None

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
        origin = np.zeros(3) if self.origin is None else np.asarray(self.origin, dtype=float).reshape(3)
        arrays = {}
        for key, value in (self.arrays or {}).items():
            value = np.asarray(value)
            if len(value) != len(sym):
                raise ValueError(f"array {key!r} has {len(value)} rows for {len(sym)} atoms")
            arrays[key] = value
        occ = None if self.occupancy is None else tuple(self.occupancy)
        if occ is not None and len(occ) != len(sym):
            raise ValueError("occupancy needs one entry per atom")
        for name, value in (("symbols", sym), ("positions", pos), ("cell", cell), ("pbc", pbc),
                            ("ids", ids), ("origin", origin), ("arrays", arrays), ("occupancy", occ)):
            object.__setattr__(self, name, value)

    def __len__(self):
        return len(self.symbols)

    def __getitem__(self, index):
        """Subset by mask or index array; keeps cell, pbc and per-atom data."""
        index = np.asarray(index)
        if index.dtype != bool:
            index = np.atleast_1d(index)
        rows = np.flatnonzero(index) if index.dtype == bool else index
        occ = None if self.occupancy is None else tuple(self.occupancy[r] for r in rows)
        return replace(self, symbols=self.symbols[index], positions=self.positions[index],
                       ids=self.ids[index], arrays={k: v[index] for k, v in self.arrays.items()},
                       occupancy=occ)

    def copy(self, **changes):
        return replace(self, **changes)

    def index_of(self, ids):
        """Row indices of the atoms with these ids."""
        return np.flatnonzero(np.isin(self.ids, np.atleast_1d(ids)))

    def concat(self, other):
        """This structure followed by other's atoms (keeps this cell)."""
        keys = set(self.arrays) & set(other.arrays)
        occ = None
        if self.occupancy is not None or other.occupancy is not None:
            occ = tuple(self.occupancy or (None,) * len(self)) + tuple(other.occupancy or (None,) * len(other))
        return replace(self, symbols=np.concatenate([self.symbols, other.symbols]),
                       positions=np.concatenate([self.positions, other.positions]),
                       ids=np.concatenate([self.ids, other.ids]),
                       arrays={k: np.concatenate([self.arrays[k], other.arrays[k]]) for k in keys},
                       occupancy=occ)


def cell_from_parameters(a, b, c, alpha, beta, gamma):
    """Lattice vectors (rows) from a, b, c (Å) and alpha, beta, gamma (degrees)."""
    al, be, ga = np.radians([alpha, beta, gamma])
    va = [a, 0.0, 0.0]
    vb = [b * np.cos(ga), b * np.sin(ga), 0.0]
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([va, vb, [cx, cy, cz]])


def _guess_element(name):
    """Element from an atom name such as 'CA', 'OW1', 'Mo12' (MDAnalysis names)."""
    letters = re.sub(r"[^A-Za-z]", "", str(name))
    for n in (2, 1):
        cand = letters[:n].capitalize()
        if cand in DATA:
            return cand
    raise ValueError(f"cannot guess an element from atom name {name!r}")


def _from_ase(obj, ids):
    if isinstance(ids, str):
        ids = obj.arrays[ids]
    arrays = {k: np.array(v) for k, v in obj.arrays.items() if k not in ("numbers", "positions")}
    calc = getattr(obj, "calc", None)
    results = getattr(calc, "results", {}) or {}
    for key in ("forces", "charges", "magmoms", "energies"):
        value = results.get(key)
        if value is not None and len(np.atleast_1d(value)) == len(obj):
            arrays.setdefault(key, np.asarray(value))
    fixed = np.zeros(len(obj), bool)
    for con in getattr(obj, "constraints", []):
        if type(con).__name__ == "FixAtoms":
            fixed[np.asarray(con.get_indices())] = True
    if fixed.any():
        arrays["fixed"] = fixed
    return Structure(obj.get_chemical_symbols(), obj.get_positions(), cell=np.asarray(obj.cell[:], float),
                     pbc=np.asarray(obj.pbc), ids=ids, arrays=arrays)


def _from_pymatgen(obj, ids):
    symbols, occupancy, mixed = [], [], False
    for site in obj.sites:
        parts = [(getattr(sp, "symbol", None) or sp.element.symbol, float(frac))
                 for sp, frac in site.species.items()]
        parts.sort(key=lambda p: -p[1])
        symbols.append(parts[0][0])
        if len(parts) > 1 or parts[0][1] < 0.999:
            mixed = True
            occupancy.append(tuple(parts))
        else:
            occupancy.append(None)
    lattice = getattr(obj, "lattice", None)
    cell = None if lattice is None else np.asarray(lattice.matrix)
    pbc = getattr(lattice, "pbc", (True, True, True)) if lattice is not None else False
    arrays = {}
    for key, values in getattr(obj, "site_properties", {}).items():
        try:
            arr = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            continue
        arrays[key] = arr
    return Structure(symbols, obj.cart_coords, cell=cell, pbc=pbc, ids=ids, arrays=arrays,
                     occupancy=tuple(occupancy) if mixed else None)


def _from_rdkit(mol, ids, conf_id=-1):
    if mol.GetNumConformers() == 0:
        raise ValueError("RDKit molecule has no 3D conformer; use inkatoms.from_smiles or embed it first")
    pos = np.asarray(mol.GetConformer(conf_id).GetPositions(), float)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    ids = np.arange(len(symbols)) if ids is None else np.asarray(ids)
    orders = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        orders[frozenset((int(ids[i]), int(ids[j])))] = float(b.GetBondTypeAsDouble())
    charges = np.array([a.GetFormalCharge() for a in mol.GetAtoms()], float)
    return Structure(symbols, pos, ids=ids, bond_orders=orders, arrays={"formal_charges": charges})


def _from_mdanalysis(obj, ids):
    ag = obj.atoms
    try:
        symbols = [str(e).capitalize() for e in ag.elements]
        if not all(s in DATA for s in symbols):
            raise ValueError
    except Exception:
        symbols = [_guess_element(n) for n in ag.names]
    dims = getattr(ag, "dimensions", None)
    cell = None if dims is None or not np.any(dims[:3]) else cell_from_parameters(*dims)
    ids = ag.indices if ids is None else ids
    return Structure(symbols, ag.positions, cell=cell, pbc=cell is not None, ids=ids)


def as_structure(obj, *, ids=None) -> Structure:
    """Convert ASE, pymatgen, RDKit or MDAnalysis objects (or ``(symbols, positions)``)
    to a Structure.

    Parameters
    ----------
    obj : Atoms, Structure, Molecule, Mol, Universe or tuple
    ids : array or str, optional
        Per-atom ids, or the name of an ASE per-atom array such as ``"uid"``.
    """
    if isinstance(obj, Structure):
        return obj if ids is None else obj.copy(ids=ids)
    if hasattr(obj, "get_chemical_symbols") and hasattr(obj, "get_positions"):
        return _from_ase(obj, ids)
    if hasattr(obj, "cart_coords") and hasattr(obj, "sites"):
        return _from_pymatgen(obj, ids)
    if hasattr(obj, "GetConformer") and hasattr(obj, "GetAtoms"):
        return _from_rdkit(obj, ids)
    if hasattr(obj, "atoms") and hasattr(getattr(obj, "atoms"), "positions") and hasattr(obj.atoms, "names"):
        return _from_mdanalysis(obj, ids)
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        symbols, positions = obj
        return Structure(symbols, positions, ids=ids)
    raise TypeError(f"cannot make a Structure from {type(obj).__name__}; pass ASE Atoms, a pymatgen "
                    "Structure/Molecule, an RDKit Mol, an MDAnalysis Universe or (symbols, positions)")


def from_smiles(smiles, *, add_hydrogens=True, seed=0, optimise=True, kekulize=True) -> Structure:
    """Build a 3D structure from SMILES with RDKit, keeping its bond orders.

    With ``kekulize=False`` aromatic bonds keep order 1.5 (drawn half dashed).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as err:
        raise ImportError("from_smiles needs RDKit: pip install rdkit") from err
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise ValueError(f"RDKit could not embed {smiles!r} in 3D")
    if optimise:
        AllChem.MMFFOptimizeMolecule(mol)
    if kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    return _from_rdkit(mol, None)
