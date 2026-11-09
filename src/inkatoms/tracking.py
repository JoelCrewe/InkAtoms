"""Follow atoms between frames when ids are missing or unreliable.

link() matches each frame to the previous one, element by element, with an
optimal assignment (as in particle tracking). The cost of linking an atom to a
track comes either from distances (to the track's last position, or to where
it would be if it kept its velocity), or from a LinkModel: a small logistic
regression trained on frames that do carry true ids, which learns how much to
trust predicted position, plain distance and changes in local bonding.

Atoms with no plausible partner within max_jump start new tracks; tracks that
find no partner are kept for `memory` frames, so an atom that briefly leaves a
cropped region (or is missed) can be picked up again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .bonds import find_bonds
from .trajectory import as_frames

_BIG = 1e6


def _coordination_vectors(s, species, cutoffs, bond_scale):
    """(N, len(species)) neighbour counts per element, a cheap local-environment signature."""
    i, j, _ = find_bonds(s, cutoffs=cutoffs, scale=bond_scale)
    out = np.zeros((len(s), len(species)))
    col = {sym: k for k, sym in enumerate(species)}
    for a, b in zip(i, j):
        out[a, col[s.symbols[b]]] += 1
        out[b, col[s.symbols[a]]] += 1
    return out


def _pair_features(prev_pos, prev_vel, prev_env, cur_pos, cur_env):
    """Features for every (track, atom) pair: distance to the predicted position,
    distance to the last position, speed implied, and change in coordination."""
    pred = prev_pos + prev_vel
    d_pred = cdist(pred, cur_pos)
    d_last = cdist(prev_pos, cur_pos)
    d_env = cdist(prev_env, cur_env, metric="cityblock")
    speed = np.linalg.norm(prev_vel, axis=1)[:, None] * np.ones_like(d_last)
    return np.stack([d_pred, d_last, d_last ** 2, d_env, speed], axis=-1)


@dataclass
class LinkModel:
    """Logistic model of P(same atom | pair features), trained on labelled frames."""

    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    names: tuple = ("distance to predicted position", "distance to last position",
                    "distance to last position, squared", "change in coordination", "speed")

    def logit(self, X):
        Z = (X - self.mean) / self.scale
        return Z @ self.weights + self.bias

    def cost(self, X):
        """-log P(match): low for likely pairs."""
        return np.logaddexp(0.0, -self.logit(X))

    def __str__(self):
        rows = [f"  {n:32s} {w:+.2f}" for n, w in zip(self.names, self.weights)]
        return "LinkModel weights (standardised features; negative = makes a match less likely):\n" + \
            "\n".join(rows)


def _features_and_labels(structs, species, cutoffs, bond_scale, stride, k_neg):
    X, y, sy = [], [], []
    envs = [_coordination_vectors(s, species, cutoffs, bond_scale) for s in structs]
    for f in range(stride, len(structs) - stride, stride):
        prev, cur, before = structs[f], structs[f + stride], structs[f - stride]
        row_before = {int(i): r for r, i in enumerate(before.ids)}
        row_cur = {int(i): r for r, i in enumerate(cur.ids)}
        for sym in set(prev.symbols):
            pa = np.flatnonzero(prev.symbols == sym)
            ca = np.flatnonzero(cur.symbols == sym)
            if len(ca) < 2:
                continue
            vel = np.array([prev.positions[r] - before.positions[row_before[int(prev.ids[r])]]
                            if int(prev.ids[r]) in row_before else np.zeros(3) for r in pa])
            F = _pair_features(prev.positions[pa], vel, envs[f][pa], cur.positions[ca], envs[f + stride][ca])
            for t_idx, r in enumerate(pa):
                true = row_cur.get(int(prev.ids[r]))
                if true is None or cur.symbols[true] != sym:
                    continue
                c_idx = int(np.flatnonzero(ca == true)[0])
                X.append(F[t_idx, c_idx])
                y.append(1)
                sy.append(sym)
                near = np.argsort(F[t_idx, :, 1])[:k_neg + 1]     # the closest wrong candidates
                for c in near:
                    if c != c_idx:
                        X.append(F[t_idx, c])
                        y.append(0)
                        sy.append(sym)
    return np.array(X), np.array(y, float), np.array(sy)


def _fit_logistic(X, y, l2, iterations):
    mean, scale = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = np.column_stack([(X - mean) / scale, np.ones(len(X))])
    w = np.zeros(Z.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        grad = Z.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        step = np.linalg.solve(H, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return LinkModel(weights=w[:-1], bias=float(w[-1]), mean=mean, scale=scale)


class LinkModels(dict):
    """One LinkModel per element (H hops further than Mo, so they need different weights)."""

    species: list = []

    def cost(self, X, symbol):
        return self[symbol].cost(X) if symbol in self else self["*"].cost(X)

    def __str__(self):
        return "\n".join(f"[{sym}] {m}" for sym, m in self.items())


def fit_link_model(frames, *, ids="uid", cutoffs=None, bond_scale=1.15, stride=1, k_neg=4, l2=1e-3,
                   iterations=50, per_element=True, min_examples=200):
    """Learn how to link atoms from frames that carry true ids (logistic regression,
    Newton iterations), one model per element unless per_element=False.
    stride: train for linking frames this far apart (match how you will use it)."""
    structs, _ = as_frames(frames, ids=ids)
    species = sorted(set().union(*[set(s.symbols) for s in structs]))
    X, y, sy = _features_and_labels(structs, species, cutoffs, bond_scale, stride, k_neg)
    if len(np.unique(y)) < 2:
        raise ValueError("need frames with true ids and more than one candidate per atom to learn from")
    models = LinkModels()
    models["*"] = _fit_logistic(X, y, l2, iterations)
    if per_element:
        for sym in np.unique(sy):
            m = sy == sym
            if m.sum() >= min_examples and len(np.unique(y[m])) == 2:
                models[str(sym)] = _fit_logistic(X[m], y[m], l2, iterations)
    models.species = species
    return models


@dataclass
class LinkResult:
    """Frames with consistent track ids (in .frames, as Structures) and a summary."""

    frames: list
    new_tracks: int = 0
    lost_tracks: int = 0
    log: list = field(default_factory=list)


def link(frames, *, method="velocity", model=None, max_jump=1.5, memory=3, cutoffs=None, bond_scale=1.15,
         ids=None):
    """Give atoms consistent ids across frames by linking each frame to the last.

    method   "distance"  cost = distance to the track's last position
             "velocity"  cost = distance to where the track would be at constant velocity
             "learned"   cost = -log P(match) from a LinkModel (fit_link_model on frames
                         with true ids, e.g. an earlier part of the same run)
    max_jump links longer than this (Å) are not allowed: the atom starts a new track
    memory   frames a track waits for its atom before it is closed
    Returns a LinkResult whose .frames are Structures with the new ids.
    """
    structs, _ = as_frames(frames, ids=ids)
    if method == "learned" and model is None:
        raise ValueError('method="learned" needs a model from fit_link_model(...)')
    species = getattr(model, "species", None) or sorted(set().union(*[set(s.symbols) for s in structs]))
    first = structs[0]
    next_id = len(first)
    out = [first.copy(ids=np.arange(len(first)))]
    # live tracks: id -> (symbol, position, velocity, env, frames missing)
    env0 = _coordination_vectors(first, species, cutoffs, bond_scale)
    tracks = {k: (first.symbols[k], first.positions[k], np.zeros(3), env0[k], 0) for k in range(len(first))}
    result = LinkResult(frames=out)
    for f in range(1, len(structs)):
        cur = structs[f]
        env = _coordination_vectors(cur, species, cutoffs, bond_scale)
        new_ids = np.full(len(cur), -1, int)
        for sym in set(cur.symbols):
            ca = np.flatnonzero(cur.symbols == sym)
            tids = [t for t, v in tracks.items() if v[0] == sym]
            if not tids:
                continue
            pos = np.array([tracks[t][1] for t in tids])
            vel = np.array([tracks[t][2] for t in tids])
            gap = np.array([tracks[t][4] + 1 for t in tids])[:, None]
            tenv = np.array([tracks[t][3] for t in tids])
            F = _pair_features(pos, vel * gap, tenv, cur.positions[ca], env[ca])
            if method == "distance":
                C = F[..., 1].copy()
            elif method == "velocity":
                C = F[..., 0].copy()
            else:
                C = model.cost(F, sym) if isinstance(model, LinkModels) else model.cost(F)
            C[F[..., 1] > max_jump * gap] = _BIG
            r, c = linear_sum_assignment(C)
            ok = C[r, c] < _BIG
            for ti, ci in zip(r[ok], c[ok]):
                new_ids[ca[ci]] = tids[ti]
        # update tracks
        seen = set()
        for row, tid in enumerate(new_ids):
            if tid < 0:
                new_ids[row] = next_id
                tid = next_id
                next_id += 1
                result.new_tracks += 1
                tracks[tid] = (cur.symbols[row], cur.positions[row], np.zeros(3), env[row], 0)
            else:
                _, last, _, _, miss = tracks[tid]
                vel = (cur.positions[row] - last) / (miss + 1)
                tracks[tid] = (cur.symbols[row], cur.positions[row], vel, env[row], 0)
            seen.add(int(tid))
        for tid in list(tracks):
            if tid not in seen:
                sym, p, v, e, miss = tracks[tid]
                if miss + 1 > memory:
                    del tracks[tid]
                    result.lost_tracks += 1
                else:
                    tracks[tid] = (sym, p, v, e, miss + 1)
        out.append(cur.copy(ids=new_ids))
    return result


def link_accuracy(linked, truth_frames, *, ids="uid"):
    """Fraction of frame-to-frame links that join the same true atom (for testing)."""
    truth, _ = as_frames(truth_frames, ids=ids)
    good = total = 0
    for f in range(1, len(truth)):
        prev_map = {int(n): int(t) for n, t in zip(linked.frames[f - 1].ids, truth[f - 1].ids)}
        for n, t in zip(linked.frames[f].ids, truth[f].ids):
            if int(n) in prev_map:
                total += 1
                good += prev_map[int(n)] == int(t)
    return good / max(total, 1)
