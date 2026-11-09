"""Legend, scale bar, colorbar, axis triad and journal figure sizes."""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from .styles import edge_of, style as get_style


def legend_handles(symbols, style="jmol", markersize=9, **style_changes):
    st = get_style(style, **style_changes)
    handles = []
    for sym in symbols:
        face = st.color(sym)
        edge = tuple(edge_of(face, st.edge_colors.get(sym, st.atom_edgecolor), st.darken))
        handles.append(Line2D([], [], ls="none", marker="o", markersize=markersize,
                              markerfacecolor=face, markeredgecolor=edge,
                              markeredgewidth=max(st.atom_edgewidth, 0.6), label=sym))
    return handles


def legend(ax, what=None, *, style=None, markersize=9, **legend_kwargs):
    """Element legend. what: the StructureArtist from draw() (uses its style and
    elements) or a list of symbols."""
    if hasattr(what, "symbols") and hasattr(what, "style"):
        symbols, st = what.symbols, style or what.style
    else:
        symbols, st = list(what or []), style or "jmol"
    legend_kwargs.setdefault("frameon", False)
    legend_kwargs.setdefault("handletextpad", 0.2)
    return ax.legend(handles=legend_handles(symbols, st, markersize), **legend_kwargs)


def scale_bar(ax, length=5.0, label=None, *, loc="lower right", color="#222222",
              thickness=None, fontsize=8, **kwargs):
    """A scale bar in Å (draw() works in Å, so data units are Å)."""
    if label is None:
        label = f"{length:g} Å"
    if thickness is None:
        thickness = 0.012 * abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    bar = AnchoredSizeBar(ax.transData, length, label, loc, frameon=False,
                          size_vertical=thickness, color=color,
                          fontproperties={"size": fontsize}, label_top=True, **kwargs)
    ax.add_artist(bar)
    return bar


def colorbar(art, ax=None, *, label=None, **kwargs):
    """Colorbar for a drawing made with color_by=."""
    if art.mappable is None:
        raise ValueError("this drawing was not coloured by a value (use color_by=)")
    kwargs.setdefault("shrink", 0.8)
    cb = art.ax.figure.colorbar(art.mappable, ax=ax or art.ax, **kwargs)
    if label:
        cb.set_label(label)
    return cb


def axes_triad(ax, art, *, loc=(0.1, 0.12), length=0.08, labels=("x", "y", "z"),
               colors=("#c0392b", "#27ae60", "#2471a3"), fontsize=7, lw=1.1):
    """Small x/y/z arrows (in axes coordinates) showing the drawing's orientation."""
    out = []
    R = art.rotation
    # draw the axis pointing most into the page first
    order = np.argsort(R[2])
    aspect = ax.get_position().width * ax.figure.get_figwidth() / \
        (ax.get_position().height * ax.figure.get_figheight())
    for k in order:
        v = R[:2, k]
        if np.hypot(*v) < 0.15:     # nearly along the line of sight: a dot
            out.append(ax.plot(*loc, "o", ms=3, color=colors[k], transform=ax.transAxes,
                               clip_on=False)[0])
            out.append(ax.text(loc[0] + 0.02, loc[1] + 0.02, labels[k], color=colors[k], fontsize=fontsize,
                               transform=ax.transAxes))
            continue
        tip = np.array(loc) + length * np.array([v[0], v[1] * aspect])
        out.append(ax.annotate("", xy=tip, xytext=loc, xycoords="axes fraction",
                               arrowprops=dict(arrowstyle="-|>", lw=lw, color=colors[k],
                                               shrinkA=0, shrinkB=0, mutation_scale=7)))
        lab = np.array(loc) + 1.35 * (tip - np.array(loc))
        out.append(ax.text(*lab, labels[k], color=colors[k], fontsize=fontsize, ha="center", va="center",
                           transform=ax.transAxes))
    return out


# Column widths (inches), checked against the publishers' guidance in Nov 2026:
#   ACS       240 pt (3.33 in) single, 7 in double (ACS journal author guidelines)
#   RSC       8.3 cm single, 17.1 cm double (rsc.org, figures and graphics)
#   Nature    89 mm single, 183 mm double (Nature research figure guide)
#   Elsevier  90 mm single, 190 mm full width (Elsevier artwork sizing)
#   AIP       3.37 in single, 6.69 in double (AIP journal guidelines)
#   APS       REVTeX column 246 pt and text 510 pt widths (Physical Review layout)
# Individual journals can differ; check the target journal before submitting.
COLUMN_WIDTHS = {
    "acs": {"single": 3.33, "double": 7.0},
    "rsc": {"single": 3.27, "double": 6.73},
    "nature": {"single": 3.50, "double": 7.20},
    "elsevier": {"single": 3.54, "double": 7.48},
    "aip": {"single": 3.37, "double": 6.69},
    "aps": {"single": 3.40, "double": 7.06},
}


def figsize(journal="acs", width="single", aspect=0.75):
    """(width, height) in inches for a journal column, e.g. figsize("rsc", "double", 0.4)."""
    try:
        w = COLUMN_WIDTHS[journal.lower()][width]
    except KeyError as err:
        raise ValueError(f"known journals: {sorted(COLUMN_WIDTHS)}; width 'single' or 'double'") from err
    return (w, w * aspect)
