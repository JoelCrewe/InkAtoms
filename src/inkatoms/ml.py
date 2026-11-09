"""Optional machine-learning extras: local-environment descriptors, and
environment states and changes along a trajectory.

Two descriptors are available:

fingerprint  built in, no extra dependencies: for each neighbouring element, a set
             of smooth radial functions (Gaussians on a grid of distances, with a
             cosine cutoff), as in Behler-Parrinello G2 symmetry functions
soap         SOAP power spectra from dscribe (pip install "inkatoms[ml]"), which also
             capture angles

Environment states come from clustering every atom's descriptor over all frames
(k-means), so "an S that stops being a mu2 bridge" shows up as a change of state
even when no single bond crosses its cutoff at that moment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

from .structure import as_structure
from .trajectory import as_frames


def fingerprints(atoms, rows=None, *, species=None, r_cut=6.0, n_shifts=12, width=0.35):
    """Radial fingerprints, shape (len(rows), len(species) * n_shifts)."""
    s = as_structure(atoms)
    rows = np.arange(len(s)) if rows is None else np.asarray(rows, int)
    species = sorted(set(s.symbols)) if species is None else list(species)
    shifts = np.linspace(0.8, r_cut - 0.5, n_shifts)
    tree = cKDTree(s.positions)
    out = np.zeros((len(rows), len(species) * n_shifts))
    for k, r in enumerate(rows):
        nb = [j for j in tree.query_ball_point(s.positions[r], r_cut) if j != r]
        if not nb:
            continue
        d = np.linalg.norm(s.positions[nb] - s.positions[r], axis=1)
        fc = 0.5 * (np.cos(np.pi * d / r_cut) + 1.0)
        g = np.exp(-((d[:, None] - shifts[None, :]) / width) ** 2) * fc[:, None]
        for e, sym in enumerate(species):
            mask = s.symbols[nb] == sym
            out[k, e * n_shifts:(e + 1) * n_shifts] = g[mask].sum(axis=0)
    return out


def soap(atoms, rows=None, *, species=None, r_cut=5.0, n_max=6, l_max=4, sigma=0.5):
    """SOAP descriptors from dscribe, shape (len(rows), n_features)."""
    try:
        from dscribe.descriptors import SOAP
    except ImportError as err:
        raise ImportError('method="soap" needs dscribe: pip install "inkatoms[ml]"') from err
    from ase import Atoms

    s = as_structure(atoms)
    rows = np.arange(len(s)) if rows is None else np.asarray(rows, int)
    species = sorted(set(s.symbols)) if species is None else list(species)
    desc = SOAP(species=species, r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma, periodic=False)
    return np.asarray(desc.create(Atoms(list(s.symbols), positions=s.positions), centers=list(rows)))


def descriptors(atoms, rows=None, *, method="fingerprint", species=None, **kwargs):
    """Local-environment descriptors with the chosen method ("fingerprint" or "soap")."""
    if method == "fingerprint":
        return fingerprints(atoms, rows, species=species, **kwargs)
    if method == "soap":
        return soap(atoms, rows, species=species, **kwargs)
    raise ValueError("method must be 'fingerprint' or 'soap'")


@dataclass
class EnvironmentStates:
    """Environment state of each tracked atom in each frame (-1 where absent)."""

    ids: list
    labels: np.ndarray        # (n_frames, n_ids) int
    centroids: np.ndarray     # (n_states, n_features), in standardised descriptor units
    times: np.ndarray | None

    def series(self, atom_id):
        return self.labels[:, self.ids.index(int(atom_id))]


def environment_states(frames, ids, *, n_states=4, method="fingerprint", id_key="uid", species=None,
                       seed=0, **kwargs):
    """Cluster the local environments of the given atoms over all frames into
    n_states states (k-means on standardised descriptors)."""
    structs, times = as_frames(frames, ids=id_key)
    ids = [int(i) for i in np.atleast_1d(ids)]
    if species is None:
        species = sorted(set().union(*[set(s.symbols) for s in structs]))
    feats, where = [], []
    for f, s in enumerate(structs):
        row_of = {int(i): r for r, i in enumerate(s.ids)}
        rows = [row_of[i] for i in ids if i in row_of]
        if rows:
            feats.append(descriptors(s, rows, method=method, species=species, **kwargs))
            where.extend((f, ids.index(int(s.ids[r]))) for r in rows)
    X = np.vstack(feats)
    # standardise, but do not blow up features that barely vary (noise would dominate)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.maximum(sd, 0.1 * sd.max() + 1e-12)
    Z = (X - mu) / sd
    k = min(n_states, len(Z))
    centroids, lab = kmeans2(Z, k, minit="++", seed=seed)
    labels = np.full((len(structs), len(ids)), -1, int)
    for (f, k_), l_ in zip(where, lab):
        labels[f, k_] = l_
    return EnvironmentStates(ids, labels, centroids, times)


@dataclass(frozen=True)
class EnvironmentEvent:
    frame: int
    atom: int
    before: int
    after: int
    time: float | None = None

    def __str__(self):
        t = f"{self.time:8.2f} ps" if self.time is not None else f"frame {self.frame:5d}"
        return f"{t}  atom {self.atom}: environment {self.before} -> {self.after}"


def environment_events(frames, ids, *, min_frames=5, states=None, **kwargs):
    """Lasting changes of environment state (see environment_states)."""
    from .events import _debounced_changes

    st = states or environment_states(frames, ids, **kwargs)
    out = []
    for k, atom in enumerate(st.ids):
        for frame, old, new in _debounced_changes(st.labels[:, k], min_frames):
            if old >= 0 and new >= 0:
                out.append(EnvironmentEvent(frame, atom, old, new,
                                            None if st.times is None else float(st.times[frame])))
    return sorted(out, key=lambda e: e.frame)
