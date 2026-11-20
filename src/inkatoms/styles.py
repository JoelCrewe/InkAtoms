"""Colour schemes and drawing styles."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Mapping

import numpy as np
from matplotlib.colors import to_rgb

from ._elements import DATA, SYMBOLS, VESTA

_ALKALI = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
_ALKALINE_EARTH = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
_NOBLE = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}
_CPK = {"H": "#ffffff", "C": "#222222", "N": "#2233ff", "O": "#ff2200",
        "F": "#1ff01f", "Cl": "#1ff01f", "Br": "#992200", "I": "#6600bb",
        "P": "#ff9900", "S": "#dddd00", "B": "#ffaa77", "Ti": "#999999",
        "Fe": "#dd7700"}


def _cpk(symbol):
    if symbol in _CPK:
        return _CPK[symbol]
    if symbol in _NOBLE:
        return "#00ffff"
    if symbol in _ALKALI:
        return "#7700ff"
    if symbol in _ALKALINE_EARTH:
        return "#007700"
    return "#dd77ff"


def _grey(symbol):
    r, g, b = to_rgb(DATA[symbol][2])
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    v = 0.18 + 0.77 * lum
    return (v, v, v)


# Okabe & Ito (2008) palette
_CB = {"H": "#ffffff", "C": "#505050", "N": "#0072b2", "O": "#d55e00", "S": "#f0e442",
       "P": "#e69f00", "F": "#009e73", "Cl": "#009e73", "Br": "#8c4a2f", "I": "#cc79a7",
       "Si": "#b3a17a", "B": "#f5b7a1", "Se": "#e3b505"}


def _colourblind(symbol):
    if symbol in _CB:
        return _CB[symbol]
    if symbol in _NOBLE:
        return "#bfe6f5"
    from .bonds import NONMETALS
    return "#56b4e9" if symbol not in NONMETALS else "#999999"


SCHEMES = {
    "jmol": lambda s: DATA[s][2],
    "vesta": lambda s: VESTA.get(s, DATA[s][2]),
    "colourblind": _colourblind,
    "cpk": _cpk,
    "grey": _grey,
    "white": lambda s: "#ffffff",
}


def scheme_colors(scheme="jmol"):
    """Every element's colour in a scheme, as {symbol: colour}."""
    return {s: SCHEMES[scheme](s) for s in SYMBOLS}


@dataclass(frozen=True)
class Style:
    """How atoms, bonds and the cell are drawn.

    Lengths are in Å (they scale with the drawing), line widths in points.
    ``"darker"`` as an edge colour means a darker shade of the face colour.
    ``fade_distance`` fades over a fixed distance behind the front atom instead of
    over the whole depth of the structure.
    """

    scheme: str = "jmol"
    colors: Mapping[str, object] = field(default_factory=dict)
    ball_scale: float = 0.27        # x van der Waals radius
    atom_edgecolor: object = "darker"
    edge_colors: Mapping[str, object] = field(default_factory=dict)
    atom_edgewidth: float = 0.6
    bond_color: object = "split"    # "split": each half in its atom's colour
    bond_width: float = 0.26
    bond_edgecolor: object = "darker"
    bond_edgewidth: float = 0.5
    darken: float = 0.45
    shading: str = "flat"           # flat, glossy or soft
    depth_fade: float = 0.45
    depth_shrink: float = 0.12
    fade_distance: float | None = None
    background: object = "white"
    highlight_edgewidth: float = 1.4
    highlight_edgecolor: object = None
    highlight_scale: float = 1.0
    cell_color: object = "#555555"
    cell_width: float = 0.05
    occlusion: float = 0.0
    dash_length: float = 0.16
    dash_gap: float = 0.11
    contact_color: object = "#6f6f6f"
    contact_width: float = 0.08
    vacancy_radius: float = 0.45

    def color(self, symbol):
        c = self.colors.get(symbol)
        return to_rgb(c if c is not None else SCHEMES[self.scheme](symbol))

    def replace(self, **changes):
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            raise TypeError(f"unknown style options: {sorted(unknown)}")
        for key in ("colors", "edge_colors"):
            if key in changes:
                changes[key] = {**getattr(self, key), **changes[key]}
        return replace(self, **changes)


PRESETS = {
    "jmol": Style(),
    "cpk": Style(scheme="cpk"),
    "greyscale": Style(scheme="grey", atom_edgecolor="#1a1a1a", bond_edgecolor="#1a1a1a",
                       atom_edgewidth=0.7),
    "bw": Style(scheme="white", atom_edgecolor="black", atom_edgewidth=0.8,
                bond_color="white", bond_edgecolor="black", bond_edgewidth=0.8,
                cell_color="black", contact_color="black"),
    "vesta": Style(scheme="vesta"),
    "colourblind": Style(scheme="colourblind", atom_edgecolor="#333333", bond_edgecolor="#333333"),
    "dark": Style(background="#121212", cell_color="#bdbdbd", contact_color="#bdbdbd",
                  atom_edgecolor="darker", darken=0.55),
}
_ALIASES = {"grayscale": "greyscale", "grey": "greyscale", "gray": "greyscale",
            "mono": "bw", "black-white": "bw", "blackwhite": "bw",
            "colorblind": "colourblind", "cb": "colourblind"}


def style(name="jmol", **changes) -> Style:
    """Return a preset Style, optionally with some fields changed.

    Presets: ``jmol`` (default), ``cpk``, ``vesta``, ``colourblind`` (Okabe-Ito colours for
    common elements), ``greyscale``, ``bw`` (black-and-white line art) and ``dark``.

    >>> style("cpk", shading="soft", colors={"Mo": "#54b5b5"})
    """
    if isinstance(name, Style):
        base = name
    else:
        key = _ALIASES.get(name.lower(), name.lower())
        if key not in PRESETS:
            raise ValueError(f"unknown style {name!r}; choose from {sorted(PRESETS)}")
        base = PRESETS[key]
    return base.replace(**changes) if changes else base


def mix(rgb, target, f):
    """Blend colours towards target by fraction f (arrays broadcast)."""
    rgb = np.asarray(rgb, float)
    return rgb + (np.asarray(target, float) - rgb) * np.asarray(f, float)[..., None]


def edge_of(face, edge, darken):
    if isinstance(edge, str) and edge == "darker":
        return np.asarray(face, float) * (1.0 - darken)
    return np.broadcast_to(np.asarray(to_rgb(edge)), np.shape(face)).astype(float)
