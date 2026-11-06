"""Colour schemes and drawing styles.

Presets
-------
jmol       Jmol element colours (S yellow, O red, N blue, C grey, H white, ...)
cpk        classic CPK model colours (C near-black, O red, N blue, S yellow, ...)
greyscale  Jmol colours mapped to grey by luminance, dark outlines; prints well
bw         black-and-white line art: white atoms, black outlines and bonds
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Mapping

import numpy as np
from matplotlib.colors import to_rgb

from ._elements import DATA, SYMBOLS

# ---------------------------------------------------------------- schemes

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


SCHEMES = {
    "jmol": lambda s: DATA[s][2],
    "cpk": _cpk,
    "grey": _grey,
    "white": lambda s: "#ffffff",
}


def scheme_colors(scheme="jmol"):
    """Every element's colour in a scheme, as {symbol: colour}."""
    return {s: SCHEMES[scheme](s) for s in SYMBOLS}


# ---------------------------------------------------------------- style

@dataclass(frozen=True)
class Style:
    """Everything about how atoms, bonds and the cell are drawn.

    Lengths marked Å are in structure units, so they scale with the picture;
    line widths are in points.
    """

    scheme: str = "jmol"                 # base element colours (jmol, cpk, grey, white)
    colors: Mapping[str, object] = field(default_factory=dict)  # per-element overrides
    ball_scale: float = 0.27             # ball radius = ball_scale x van der Waals radius
    atom_edgecolor: object = "darker"    # "darker" (of the face colour) or a colour
    atom_edgewidth: float = 0.6          # pt
    bond_color: object = "split"         # "split" (each half takes its atom's colour) or a colour
    bond_width: float = 0.26             # Å
    bond_edgecolor: object = "darker"
    bond_edgewidth: float = 0.5          # pt
    darken: float = 0.45                 # how much darker "darker" edges are (0-1)
    shading: str = "flat"                # flat | glossy | soft
    depth_fade: float = 0.45             # how far the back fades towards the background (0-1)
    depth_shrink: float = 0.12           # how much smaller the back is drawn (0-1)
    background: object = "white"         # colour faded towards
    highlight_edgewidth: float = 1.4     # pt, for highlighted atoms
    highlight_scale: float = 1.0         # radius factor for highlighted atoms
    cell_color: object = "#555555"
    cell_width: float = 0.05             # Å

    def color(self, symbol):
        c = self.colors.get(symbol)
        return to_rgb(c if c is not None else SCHEMES[self.scheme](symbol))

    def replace(self, **changes):
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            raise TypeError(f"unknown style options: {sorted(unknown)}")
        if "colors" in changes:
            changes["colors"] = {**self.colors, **changes["colors"]}
        return replace(self, **changes)


PRESETS = {
    "jmol": Style(),
    "cpk": Style(scheme="cpk"),
    "greyscale": Style(scheme="grey", atom_edgecolor="#1a1a1a", bond_edgecolor="#1a1a1a",
                       atom_edgewidth=0.7),
    "bw": Style(scheme="white", atom_edgecolor="black", atom_edgewidth=0.8,
                bond_color="white", bond_edgecolor="black", bond_edgewidth=0.8,
                cell_color="black"),
}
_ALIASES = {"grayscale": "greyscale", "grey": "greyscale", "gray": "greyscale",
            "mono": "bw", "black-white": "bw", "blackwhite": "bw"}


def style(name="jmol", **changes) -> Style:
    """A preset Style by name (jmol, cpk, greyscale, bw), with any fields changed.

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


# ---------------------------------------------------------------- colour maths

def mix(rgb, target, f):
    """Blend colours towards target by fraction f (arrays broadcast)."""
    rgb = np.asarray(rgb, float)
    return rgb + (np.asarray(target, float) - rgb) * np.asarray(f, float)[..., None]


def edge_of(face, edge, darken):
    if isinstance(edge, str) and edge == "darker":
        return np.asarray(face, float) * (1.0 - darken)
    return np.broadcast_to(np.asarray(to_rgb(edge)), np.shape(face)).astype(float)
