"""Find bond-forming and bond-breaking events in a trajectory, group them into
reactions, and draw a reaction as a strip of labelled panels.

The rule is simple and explainable: two atoms are bonded when they are within
the bond cutoff (the same cutoffs draw() uses), and a change only counts once
the new state has lasted min_frames frames, so vibration across the cutoff is
not reported as chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._elements import DATA
from .bonds import find_bonds
from .trajectory import as_frames

_ORDER = list(DATA)


def _pair_label(sa, sb):
    a, b = sorted((sa, sb), key=lambda s: -_ORDER.index(s))    # heavier element first
    return f"{a}–{b}"


@dataclass(frozen=True)
class BondEvent:
    frame: int                 # first frame of the new state
    kind: str                  # "form" or "break"
    pair: tuple                # (id_a, id_b)
    symbols: tuple             # (symbol_a, symbol_b)
    time: float | None = None  # ps, when the frames carry times
    note: str = ""             # "atom added" / "removed" when an atom appears or leaves bonded
                               # (insertions and deletions in grand-canonical runs)

    @property
    def label(self):
        text = f"{_pair_label(*self.symbols)} {'forms' if self.kind == 'form' else 'breaks'}"
        return f"{text} ({self.note})" if self.note else text

    def __str__(self):
        t = f"{self.time:8.2f} ps" if self.time is not None else f"frame {self.frame:5d}"
        return f"{t}  {self.label}  ({self.pair[0]}–{self.pair[1]})"


@dataclass
class Reaction:
    """Events that share atoms, in time order."""

    events: list = field(default_factory=list)

    @property
    def atoms(self):
        return sorted({i for e in self.events for i in e.pair})

    @property
    def first(self):
        return min(e.frame for e in self.events)

    @property
    def last(self):
        return max(e.frame for e in self.events)

    def __str__(self):
        return "\n".join(str(e) for e in sorted(self.events, key=lambda e: e.frame))


def bond_states(frames, *, ids="uid", cutoffs=None, bond_scale=1.15, pbc=None, only=None, pairs=None):
    """Bonded state of atom pairs in every frame.

    Returns (pairs, states, symbols, structs, times): states[f, k] is 1 when
    pair k is bonded in frame f, 0 when both atoms are present but not bonded,
    -1 when either is missing. Only pairs bonded in at least one frame are
    kept; only= restricts to pairs involving these ids, pairs= to element pairs
    such as {"Mo-S", "S-H"}.
    """
    structs, times = as_frames(frames, ids=ids)
    only = None if only is None else set(int(i) for i in np.atleast_1d(only))
    want = None if pairs is None else {frozenset(p.replace(" ", "").split("-")) for p in pairs}
    bonded = []
    sym_of = {}
    for s in structs:
        i, j, _ = find_bonds(s, cutoffs=cutoffs, scale=bond_scale, pbc=pbc)
        a, b = s.ids[i], s.ids[j]
        sym_of.update(zip(map(int, s.ids), s.symbols))
        frame_pairs = set()
        for x, y, sx, sy in zip(a, b, s.symbols[i], s.symbols[j]):
            if only is not None and int(x) not in only and int(y) not in only:
                continue
            if want is not None and frozenset((sx, sy)) not in want:
                continue
            frame_pairs.add((min(int(x), int(y)), max(int(x), int(y))))
        bonded.append(frame_pairs)
    all_pairs = sorted(set().union(*bonded)) if bonded else []
    present = [set(map(int, s.ids)) for s in structs]
    states = np.zeros((len(structs), len(all_pairs)), int)
    for f, (fp, pres) in enumerate(zip(bonded, present)):
        for k, (x, y) in enumerate(all_pairs):
            states[f, k] = 1 if (x, y) in fp else (0 if (x in pres and y in pres) else -1)
    symbols = [(sym_of[x], sym_of[y]) for x, y in all_pairs]
    return all_pairs, states, symbols, structs, times


def _debounced_changes(series, min_frames):
    """(frame, old, new) for every change of a 1 (bonded) / 0 (apart) / -1
    (an atom missing) series whose new state lasts at least min_frames frames."""
    out = []
    current = series[0]
    for k in range(1, len(series)):
        if series[k] == current:
            continue
        run = series[k:k + min_frames]
        if len(run) == min_frames and np.all(run == series[k]):
            out.append((k, int(current), int(series[k])))
            current = series[k]
    return out


def find_events(frames, *, ids="uid", cutoffs=None, bond_scale=1.15, min_frames=5, only=None,
                pairs=None, pbc=None):
    """Bond-forming and bond-breaking events in a trajectory.

    frames      ASE frames (e.g. ase.io.read(path, ":")) or Structures
    ids         per-atom id array that persists between frames (default "uid")
    cutoffs     pair cutoffs in Å as in draw(), e.g. {"Mo-S": 2.85, "S-H": 1.8}
    min_frames  a new state must last this many frames to count
    only        report only events involving these ids
    pairs       report only these element pairs, e.g. {"Mo-S", "S-H"}
    """
    all_pairs, states, symbols, _, times = bond_states(frames, ids=ids, cutoffs=cutoffs,
                                                        bond_scale=bond_scale, pbc=pbc, only=only, pairs=pairs)
    events = []
    for k, pair in enumerate(all_pairs):
        for frame, old, new in _debounced_changes(states[:, k], min_frames):
            if new == 1:                          # apart -> bonded, or arrives bonded
                kind, note = "form", "atom added" if old == -1 else ""
            elif old == 1:                        # bonded -> apart, or leaves bonded
                kind, note = "break", "removed" if new == -1 else ""
            else:
                continue                          # apart <-> missing: nothing happened to the bond
            events.append(BondEvent(frame=frame, kind=kind, pair=pair, symbols=symbols[k],
                                    time=None if times is None else float(times[frame]), note=note))
    return sorted(events, key=lambda e: (e.frame, e.pair))


def group_reactions(events, *, window=None):
    """Group events that share an atom (within window frames of each other, if
    given) into Reactions, largest first."""
    events = sorted(events, key=lambda e: e.frame)
    parent = list(range(len(events)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    last_by_atom = {}
    for k, e in enumerate(events):
        for atom in e.pair:
            for other in last_by_atom.get(atom, []):
                if window is None or e.frame - events[other].frame <= window:
                    parent[root(k)] = root(other)
            last_by_atom.setdefault(atom, []).append(k)
    groups = {}
    for k in range(len(events)):
        groups.setdefault(root(k), []).append(events[k])
    return sorted((Reaction(sorted(g, key=lambda e: e.frame)) for g in groups.values()),
                  key=lambda r: (-len(r.events), r.first))


def find_reactions(frames, **kwargs):
    """find_events() then group_reactions(); window= is passed to the grouping."""
    window = kwargs.pop("window", None)
    return group_reactions(find_events(frames, **kwargs), window=window)


def _pick_frames(reaction, n_panels, lead, settle, n_frames):
    """Frames for a strip: one before the reaction, then one per event (just after
    it, so the new state shows); thinned evenly if there are more events than panels."""
    events = sorted(reaction.events, key=lambda e: e.frame)
    steps = []
    for e in events:          # events in the same frame share a panel
        if steps and steps[-1][0] == e.frame:
            steps[-1][1].append(e)
        else:
            steps.append((e.frame, [e]))
    if n_panels is not None and len(steps) > n_panels - 1:
        keep = np.unique(np.round(np.linspace(0, len(steps) - 1, n_panels - 1)).astype(int))
        steps = [steps[k] for k in keep]
    start = max(0, events[0].frame - lead)
    picks = [(start, [])]
    for f, evs in steps:
        if all(e.note == "removed" for e in evs):
            picks.append((max(f - 1, 0), evs))          # the last frame before the atoms vanish
        else:
            picks.append((min(f + settle, n_frames - 1), evs))
    return picks


def event_strip(axes, frames, reaction, *, ids="uid", around=7.0, lead=10, settle=2, view="z",
                time_unit="ps", labels=True, show_broken=True, highlight=True, fontsize=8,
                **draw_kwargs):
    """Draw a reaction as a strip of panels: the state before it, then each step.

    axes        a list/array of Axes (one panel per step plus one for the start)
    around      crop each frame to atoms within this distance (Å) of the reacting atoms
    lead        frames before the first event for the starting panel
    settle      frames after each event to show its new state
    show_broken draw bonds that just broke as dashed contacts
    Other keyword arguments go to draw() (style, cutoffs, depth_cue, ...); the same
    view, depth fade and framing are used in every panel.
    Returns (artists, frame_indices).
    """
    from .render import draw
    from . import select

    axes = list(np.ravel(axes))
    structs, times = as_frames(frames, ids=ids)
    picks = _pick_frames(reaction, len(axes), lead, settle, len(structs))[:len(axes)]
    atoms = reaction.atoms
    crops = []
    for f, _ in picks:
        s = structs[f]
        present = [a for a in atoms if a in set(map(int, s.ids))]
        crops.append(select.around(s, at=present, radius=around, keep=present) if present else s)

    from .view import auto_view, view_matrix
    v = auto_view(crops[0].positions) if isinstance(view, str) and view == "auto" else view
    R = view_matrix(v, up=draw_kwargs.pop("up", None), rotate=draw_kwargs.get("rotate"))
    if "depth_range" not in draw_kwargs:
        d = np.concatenate([(c.positions @ R.T)[:, 2] for c in crops])
        draw_kwargs["depth_range"] = (float(d.min()), float(d.max()))
    arts = []
    for ax, (f, evs), crop in zip(axes, picks, crops):
        kw = dict(draw_kwargs)
        if highlight:
            kw.setdefault("highlight", [a for a in atoms if a in set(map(int, crop.ids))])
            kw.setdefault("highlight_front", True)
        if show_broken:
            broken = [e.pair for e in evs if e.kind == "break"
                      and all(p in set(map(int, crop.ids)) for p in e.pair)]
            if broken:
                kw["contacts"] = broken
        arts.append(draw(ax, crop, view=v, **kw))
    for ax in axes[len(arts):]:          # more panels than steps: hide the spares
        ax.set_visible(False)
    b = np.array([a.bounds for a in arts])
    pad = draw_kwargs.get("pad", 0.5)
    for ax in axes[:len(arts)]:
        ax.set_xlim(b[:, 0].min() - pad, b[:, 1].max() + pad)
        ax.set_ylim(b[:, 2].min() - pad, b[:, 3].max() + pad)
    if labels:
        for k, (ax, (f, evs)) in enumerate(zip(axes, picks)):
            title = "before" if not evs else "\n".join(dict.fromkeys(e.label for e in evs))
            ax.set_title(f"({_roman(k + 1)}) {title}", fontsize=fontsize, loc="left")
            t = f"{times[f]:.1f} {time_unit}" if times is not None else f"frame {f}"
            ax.text(0.98, 0.98, t, transform=ax.transAxes, ha="right", va="top", fontsize=fontsize - 1,
                    color="#777777")
    return arts, [f for f, _ in picks]


def _roman(k):
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while k >= v:
            out += s
            k -= v
    return out
