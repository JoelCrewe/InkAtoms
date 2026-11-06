"""Legend and scale bar."""

from __future__ import annotations

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from .styles import edge_of, style as get_style


def legend_handles(symbols, style="jmol", markersize=9, **style_changes):
    st = get_style(style, **style_changes)
    handles = []
    for sym in symbols:
        face = st.color(sym)
        edge = tuple(edge_of(face, st.atom_edgecolor, st.darken))
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
