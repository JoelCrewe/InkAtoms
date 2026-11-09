"""Tests for trajectories, events, trails, comparison and the ML extras,
on small synthetic trajectories with known answers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import inkatoms as ia
from inkatoms import ml
from inkatoms.events import _debounced_changes

ase = pytest.importorskip("ase")
import ase.build  # noqa: E402
import ase.cluster  # noqa: E402
from ase import Atoms  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def dissociation(n=40, jitter=0.0, seed=0):
    """An S2 pair pulled apart half way through, plus a spectator H that wanders,
    and a second H that is inserted bonded to the first S at frame 30."""
    rng = np.random.default_rng(seed)
    frames = []
    for f in range(n):
        d = 2.0 if f < n // 2 else 3.4
        pos = [[0, 0, 0], [d, 0, 0], [0, 8 + 0.1 * f, 0]]
        sym, uid = "SSH", [1, 2, 3]
        if f >= 30:
            pos.append([-1.3, 0, 0])
            sym, uid = sym + "H", uid + [4]
        pos = np.array(pos, float) + rng.normal(0, jitter, (len(pos), 3))
        a = Atoms(sym, positions=pos)
        a.new_array("uid", np.array(uid))
        a.info["time_ps"] = 0.1 * f
        frames.append(a)
    return frames


# ---------------------------------------------------------------- trajectory helpers

def test_as_frames_track_and_mobile():
    frames = dissociation()
    structs, times = ia.as_frames(frames, ids="uid")
    assert len(structs) == 40 and np.isclose(times[-1], 3.9)
    path = ia.track(structs, [3, 4])
    assert np.isfinite(path[:, 0]).all() and np.isnan(path[:30, 1]).all()
    assert ia.mobile(frames, n=1)[0] == 3          # the wandering H moves furthest


def test_unwrap_removes_jumps():
    cell = np.eye(3) * 10
    path = np.array([[9.5, 0, 0], [9.9, 0, 0], [0.3, 0, 0], [0.7, 0, 0]])
    un = ia.unwrap(path, cell)
    assert np.allclose(np.diff(un[:, 0]), 0.4, atol=1e-9)


# ---------------------------------------------------------------- events

def test_debounce_ignores_flicker():
    s = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
    assert _debounced_changes(s, 3) == [(6, 1, 0)]


def test_find_events_break_and_insertion():
    ev = ia.find_events(dissociation(jitter=0.02), cutoffs={"S-S": 2.4, "S-H": 1.8}, min_frames=3)
    kinds = [(e.kind, e.pair, e.note) for e in ev]
    assert ("break", (1, 2), "") in kinds
    assert ("form", (1, 4), "atom added") in kinds
    brk = next(e for e in ev if e.kind == "break")
    assert brk.frame == 20 and np.isclose(brk.time, 2.0) and brk.label == "S–S breaks"


def test_group_reactions_and_strip():
    frames = dissociation()
    rx = ia.find_reactions(frames, cutoffs={"S-S": 2.4, "S-H": 1.8}, min_frames=3)
    assert rx[0].atoms == [1, 2, 4] and len(rx[0].events) == 2
    fig, axes = plt.subplots(1, 3)
    arts, picked = ia.event_strip(axes, frames, rx[0], around=10, cutoffs={"S-S": 2.4, "S-H": 1.8})
    assert len(arts) == 3 and picked[0] < 20 <= picked[1]
    assert axes[1].get_title("left").endswith("S–S breaks")
    assert len({ax.get_xlim() for ax in axes}) == 1


# ---------------------------------------------------------------- trails

def test_trails_add_segments():
    frames = dissociation()
    fig, ax = plt.subplots()
    last = ia.as_structure(frames[-1], ids="uid")
    plain = len(ia.draw(ax, last, view="z").collection.get_paths())
    art = ia.draw(ax, last, view="z", trails={"frames": frames, "ids": [3], "smooth": 1})
    assert len(art.collection.get_paths()) == plain + 39
    art = ia.draw(ax, last, view="z", trails={"frames": frames, "ids": "mobile", "n": 1, "color": "time"})
    assert len(art.collection.get_paths()) > plain


# ---------------------------------------------------------------- comparison

def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(10, 3))
    t = np.radians(40)
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    Q = P @ R.T + [1, 2, 3]
    R2, t2 = ia.kabsch(P, Q)
    assert np.allclose(R2, R) and np.allclose(t2, [1, 2, 3])


def test_match_without_ids_and_compare():
    from ase.cluster import Icosahedron

    a = ia.as_structure(Icosahedron("Cu", 2))
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(a))
    b = a.copy(positions=a.positions + rng.normal(0, 0.1, a.positions.shape))[perm]
    b = b.copy(ids=np.arange(len(b)) + 500)
    for method in ("distance", "hybrid"):
        ra, rb = ia.match(a, b, method=method)
        assert all(perm[y] == x for x, y in zip(ra, rb))
    c = ia.compare(a, b)                     # auto -> distance (no shared ids)
    assert c.method == "distance" and c.rmsd < 0.25
    # a rigid rotation is removed by the alignment
    t = np.radians(30)
    R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    assert ia.compare(a, a.copy(positions=a.positions @ R.T)).rmsd < 1e-9


def test_draw_comparison():
    a = ia.as_structure(ase.build.molecule("CH3CH2OH"))
    b = a.copy(positions=a.positions + np.r_[[[0.5, 0, 0]], np.zeros((len(a) - 1, 3))])
    fig, ax = plt.subplots()
    art, cmp_ = ia.draw_comparison(ax, a, b, min_shift=0.2, align=False)
    assert cmp_.largest(1)[0][0] == 0 and art.mappable is not None
    assert len(art.structure) == 2 * len(a)
    assert np.isclose(art.structure.arrays["_fade"][len(a):], 0.65).all()


# ---------------------------------------------------------------- ML extras

def test_fingerprints_and_environment_events():
    frames = dissociation()
    fp = ml.fingerprints(ia.as_structure(frames[0]))
    assert fp.shape == (3, 2 * 12) and fp[2].sum() < fp[0].sum()   # the far H sees little
    st = ml.environment_states(frames, [1], n_states=2)
    assert set(np.unique(st.series(1))) == {0, 1}
    ev = ml.environment_events(frames, [1], states=st, min_frames=3)
    assert len(ev) >= 1 and ev[0].frame == 20


def test_soap_if_available():
    pytest.importorskip("dscribe")
    d = ml.soap(ia.as_structure(ase.build.molecule("H2O")))
    assert d.shape[0] == 3 and d.shape[1] > 10
