"""Find bond-forming and bond-breaking events in a trajectory, group them into
reactions, and draw a reaction as a strip of labelled panels.

The rule is simple and explainable: two atoms are bonded when they are within
the bond cutoff (the same cutoffs draw() uses), and a change only counts once
the new state has lasted min_frames frames, so vibration across the cutoff is
not reported as chemistry.
"""

from __future__ import annotations

import textwrap
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


def active_colour(colour):
    """A saturated, darker version of an element colour, for atoms that take part;
    near-white (H) becomes red, as in the MoS3 paper figures."""
    import colorsys

    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(colour)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.15:                     # white / grey: no hue to deepen
        return (0.91, 0.31, 0.23)
    return colorsys.hsv_to_rgb(h, min(1.0, s * 1.35 + 0.1), v * 0.78)


def event_strip(axes, frames, reaction, *, ids="uid", focus=None, crop="slab", slab=4.0, window=None,
                anchor=(0.5, 0.5),
                around=7.0, lead=10, settle=2, view="z", roles=True, role_colors=None, notes=False,
                time_unit="ps", labels=True, show_broken=True, fontsize=8, wrap=18, **draw_kwargs):
    """Draw a reaction as a strip of panels: the state before it, then each step.

    axes        a list/array of Axes (one panel per step plus one for the start)
    focus       the atom the reaction is about (default: the one in most events)
    crop        "slab" (default): a slab slab Å either side of the focus atom's starting
                position, the same window in every panel, as in a publication strip;
                "around": atoms within around Å of the reacting atoms, framed together
    window      (width, height) of the slab window in Å (default: fit the panels)
    anchor      where the focus atom's starting position sits in the window, as a
                fraction of its width and height, e.g. (0.5, 0.33) leaves room above
    roles       colour the focus atom and whatever is bonded to it in each frame in
                saturated versions of their element colours (role_colors overrides,
                e.g. {"S": "#e0a01c", "Mo": "#2f7f92", "H": "#e8503a"})
    notes       show "(atom added)" / "(removed)" in panel titles
    lead, settle  frames before the first event / after each event to show
    show_broken draw bonds that just broke as dashed contacts
    wrap        wrap panel titles at this many characters (None: no wrapping)
    Other keyword arguments go to draw() (style, cutoffs, colors, depth_cue, ...).
    Returns (artists, frame_indices).
    """
    from . import select
    from .render import draw
    from .styles import style as get_style
    from .view import auto_view, view_matrix

    axes = list(np.ravel(axes))
    structs, times = as_frames(frames, ids=ids)
    picks = _pick_frames(reaction, len(axes), lead, settle, len(structs))[:len(axes)]
    atoms = reaction.atoms
    if focus is None:
        counts = {}
        for e in reaction.events:
            for i in e.pair:
                counts[i] = counts.get(i, 0) + 1
        focus = max(counts, key=counts.get)
    first = structs[picks[0][0]]
    v = auto_view(first.positions) if isinstance(view, str) and view == "auto" else view
    up = draw_kwargs.pop("up", None)
    R = view_matrix(v, up=up, rotate=draw_kwargs.get("rotate"))
    start = first.positions[first.index_of(focus)[0]] @ R.T          # page x, y, depth of the site

    crops = []
    for f, _ in picks:
        s = structs[f]
        present = [a for a in atoms if a in set(map(int, s.ids))]
        if crop == "slab":
            q = s.positions @ R.T
            keep = np.abs(q[:, 2] - start[2]) <= slab
            if window is not None:
                x0 = start[0] - anchor[0] * window[0]
                y0 = start[1] - anchor[1] * window[1]
                keep &= (q[:, 0] >= x0 - 2) & (q[:, 0] <= x0 + window[0] + 2) & \
                        (q[:, 1] >= y0 - 2) & (q[:, 1] <= y0 + window[1] + 2)
            keep |= np.isin(s.ids, present)
            crops.append(s[keep].copy(pbc=False))
        elif crop == "around":
            crops.append(select.around(s, at=present, radius=around, keep=present) if present else s)
        else:
            raise ValueError("crop must be 'slab' or 'around'")
    if "depth_range" not in draw_kwargs:
        if crop == "slab":
            draw_kwargs["depth_range"] = (start[2] - slab, start[2] + slab)
        else:
            d = np.concatenate([(c.positions @ R.T)[:, 2] for c in crops])
            draw_kwargs["depth_range"] = (float(d.min()), float(d.max()))
    draw_kwargs.setdefault("depth_fade", 0.8)
    st = get_style(draw_kwargs.get("style", "jmol"), **{k: draw_kwargs[k] for k in ("colors",) if k in draw_kwargs})

    arts = []
    for ax, (f, evs), crop_s in zip(axes, picks, crops):
        kw = dict(draw_kwargs)
        if roles:
            # the focus atom and its bonded partners in this frame, in "active" colours
            i, j, _ = find_bonds(crop_s, cutoffs=kw.get("cutoffs"), scale=kw.get("bond_scale", 1.15))
            a_ids, b_ids = crop_s.ids[i], crop_s.ids[j]
            partners = set(b_ids[a_ids == focus]) | set(a_ids[b_ids == focus])
            hl = {}
            for atom in [focus, *sorted(map(int, partners))]:
                rows = crop_s.index_of(atom)
                if len(rows):
                    sym = crop_s.symbols[rows[0]]
                    hl[atom] = (role_colors or {}).get(sym) or active_colour(st.color(sym))
            kw.setdefault("highlight", hl)
            kw.setdefault("highlight_front", True)
        if show_broken:
            broken = [e.pair for e in evs if e.kind == "break"
                      and all(p in set(map(int, crop_s.ids)) for p in e.pair)]
            if broken:
                kw["contacts"] = broken
        arts.append(draw(ax, crop_s, view=v, up=up, autoscale=False, **kw))
    for ax in axes[len(arts):]:          # more panels than steps: hide the spares
        ax.set_visible(False)
    if crop == "slab" and window is not None:
        x0, y0 = start[0] - anchor[0] * window[0], start[1] - anchor[1] * window[1]
        xl, yl = (x0, x0 + window[0]), (y0, y0 + window[1])
    else:
        b = np.array([a.bounds for a in arts])
        pad = draw_kwargs.get("pad", 0.5)
        xl = (b[:, 0].min() - pad, b[:, 1].max() + pad)
        yl = (b[:, 2].min() - pad, b[:, 3].max() + pad)
    for ax in axes[:len(arts)]:
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
    if labels:
        for k, (ax, (f, evs)) in enumerate(zip(axes, picks)):
            names = ["before"] if not evs else list(dict.fromkeys(
                e.label if notes else e.label.split(" (")[0] for e in evs))
            if wrap:
                names = [textwrap.fill(x, wrap) for x in names]
            ax.set_title(f"({_roman(k + 1)}) " + "\n".join(names), fontsize=fontsize, loc="left")
            tt = f"{times[f]:.1f} {time_unit}" if times is not None else f"frame {f}"
            ax.text(0.98, 0.98, tt, transform=ax.transAxes, ha="right", va="top", fontsize=fontsize - 1,
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


def neighbour_counts(frames, atom_id, *, ids="uid", cutoffs=None, bond_scale=1.15):
    """How many neighbours of each element one atom has in every frame.

    Returns (times or frame numbers, {symbol: counts}); counts are NaN in frames
    where the atom is absent.
    """
    structs, times = as_frames(frames, ids=ids)
    per = []
    for s in structs:
        rows = s.index_of(atom_id)
        if len(rows) == 0:
            per.append(None)
            continue
        i, j, _ = find_bonds(s, cutoffs=cutoffs, scale=bond_scale)
        r = rows[0]
        partners = np.concatenate([j[i == r], i[j == r]])
        per.append(dict(zip(*np.unique(s.symbols[partners], return_counts=True))))
    symbols = sorted({k for d in per if d for k in d}, key=lambda x: -_ORDER.index(x))
    counts = {sym: np.array([np.nan if d is None else d.get(sym, 0) for d in per], float) for sym in symbols}
    t = times if times is not None else np.arange(len(structs), dtype=float)
    return t, counts


def plot_neighbours(ax, frames, atom_id, *, ids="uid", cutoffs=None, bond_scale=1.15, events=None,
                    colors=None, style="jmol", label_events=True, fontsize=7, min_frames=1):
    """Step plot of an atom's neighbour count per element over time, with bond
    events marked: a readable timeline of what happened to it."""
    from .styles import style as get_style

    st = get_style(style)
    t, counts = neighbour_counts(frames, atom_id, ids=ids, cutoffs=cutoffs, bond_scale=bond_scale)
    lines = []
    if min_frames > 1:                       # hide flicker shorter than min_frames
        for sym, n in counts.items():
            series = np.where(np.isfinite(n), n, -1).astype(int)
            clean = np.full(len(series), series[0], float)
            for f, _, new in _debounced_changes(series, min_frames):
                clean[f:] = new
            clean[clean < 0] = np.nan
            counts[sym] = clean
    for k, (sym, n) in enumerate(counts.items()):
        colour = (colors or {}).get(sym) or _darker(st.color(sym))
        lines.append(ax.step(t, n + 0.06 * (k - (len(counts) - 1) / 2), where="post", color=colour, lw=1.4,
                             label=f"{sym} neighbours")[0])
    for e in events or []:
        x = e.time if e.time is not None else e.frame
        ax.axvline(x, color="#999999", lw=0.6, ls=(0, (2, 2)), zorder=0)
    top = max(np.nanmax(n) for n in counts.values()) if counts else 1
    ax.set_ylim(-0.4, top + 0.8)
    ax.set_yticks(range(int(top) + 1))
    ax.set_ylabel("neighbours", fontsize=fontsize + 0.5)
    ax.set_xlabel("time / ps" if not np.allclose(t, np.arange(len(t))) else "frame", fontsize=fontsize + 0.5)
    ax.legend(handles=lines, fontsize=fontsize, frameon=False, loc="upper right")
    return lines


def _darker(rgb):
    """Light element colours (S yellow, H white) need darkening to read as lines."""
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    f = 0.55 if lum > 0.75 else 0.85
    return (r * f, g * f, b * f)
