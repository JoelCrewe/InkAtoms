"""inkatoms: elegant, vector ball-and-stick drawings of atomic structures in matplotlib.

    import matplotlib.pyplot as plt
    import inkatoms as ia

    fig, ax = plt.subplots()
    art = ia.draw(ax, atoms, view="-y", style="jmol", highlight=[12])
    ia.legend(ax, art)
"""

from . import select
from .bonds import coordination, cutoff_matrix, find_bonds
from .extras import legend, legend_handles, scale_bar
from .render import StructureArtist, draw
from .structure import Structure, as_structure
from .styles import PRESETS, Style, scheme_colors, style
from .view import project, view_matrix

__version__ = "0.1.0"

__all__ = [
    "draw", "StructureArtist", "Structure", "as_structure", "Style", "style", "PRESETS",
    "scheme_colors", "find_bonds", "coordination", "cutoff_matrix", "legend",
    "legend_handles", "scale_bar", "select", "view_matrix", "project",
]
