"""Frames, times and per-atom tracks."""

from __future__ import annotations

import numpy as np

from .structure import as_structure

_TIME_KEYS = (("time_ps", 1.0), ("time_fs", 1e-3), ("time", 1.0), ("t", 1.0))


def frame_time(frame):
    """Time of an ASE frame in ps from its info (time_ps / time_fs / time), or None."""
    info = getattr(frame, "info", None) or {}
    for key, factor in _TIME_KEYS:
        if key in info:
            try:
                return float(info[key]) * factor
            except (TypeError, ValueError):
                pass
    return None


def as_frames(frames, ids=None):
    """(list of Structures, times in ps or None) from ASE frames / Structures.

    ids: per-atom id array name (e.g. "uid") so atoms can be followed between
    frames even when atoms are added, removed or reordered.
    """
    from .structure import Structure

    def one(f):
        if isinstance(f, Structure):
            return f
        if isinstance(ids, str) and ids not in getattr(f, "arrays", {}):
            return as_structure(f)          # no id array: use row order
        return as_structure(f, ids=ids)

    frames = list(frames)
    structs = [one(f) for f in frames]
    times = [frame_time(f) for f in frames]
    return structs, (np.array(times, float) if all(t is not None for t in times) else None)


def track(structs, ids):
    """Positions of the given ids in every frame, shape (n_frames, n_ids, 3);
    NaN where an atom is absent from a frame."""
    ids = list(np.atleast_1d(ids))
    out = np.full((len(structs), len(ids), 3), np.nan)
    for f, s in enumerate(structs):
        row_of = {int(i): r for r, i in enumerate(s.ids)}
        for k, atom_id in enumerate(ids):
            r = row_of.get(int(atom_id))
            if r is not None:
                out[f, k] = s.positions[r]
    return out


def unwrap(path, cell=None, pbc=(True, True, True)):
    """Remove periodic jumps from a (n_frames, 3) or (n_frames, n_ids, 3) path."""
    path = np.array(path, float)
    if cell is None:
        return path
    cell = np.asarray(cell, float)
    inv = np.linalg.inv(cell)
    pbc = np.asarray(pbc, bool)
    frac = path @ inv
    step = np.diff(frac, axis=0)
    step[..., pbc] -= np.round(step[..., pbc])
    step = np.nan_to_num(step)
    frac[1:] = frac[0] + np.cumsum(step, axis=0)
    return frac @ cell


def smooth(path, window=1):
    """Moving average along time (ignoring NaN gaps); window in frames."""
    path = np.asarray(path, float)
    if window <= 1:
        return path
    out = np.full_like(path, np.nan)
    half = window // 2
    for f in range(len(path)):
        seg = path[max(0, f - half):f + half + 1]
        if np.isfinite(path[f]).all():
            out[f] = np.nanmean(seg, axis=0)
    return out


def mobile(frames, n=10, *, symbols=None, ids="uid", among=None, metric="displacement"):
    """Ids of the n atoms that move most: net displacement between their first
    and last appearance ("displacement") or total path length ("path").
    among: only consider these ids (e.g. the atoms of the drawn crop)."""
    structs, _ = as_frames(frames, ids=ids)
    all_ids = sorted(set().union(*[set(map(int, s.ids)) for s in structs]))
    if among is not None:
        allowed = set(int(i) for i in among)
        all_ids = [i for i in all_ids if i in allowed]
    if symbols is not None:
        keep = set(np.atleast_1d(symbols))
        sym_of = {}
        for s in structs:
            sym_of.update(zip(map(int, s.ids), s.symbols))
        all_ids = [i for i in all_ids if sym_of[i] in keep]
    paths = track(structs, all_ids)
    score = np.zeros(len(all_ids))
    for k in range(len(all_ids)):
        p = paths[:, k][np.isfinite(paths[:, k]).all(axis=1)]
        if len(p) < 2:
            continue
        score[k] = np.linalg.norm(p[-1] - p[0]) if metric == "displacement" else \
            np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    order = np.argsort(score)[::-1][:n]
    return [all_ids[k] for k in order]
