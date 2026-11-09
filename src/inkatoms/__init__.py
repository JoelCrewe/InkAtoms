"""inkatoms: elegant, vector drawings of atomic structures in matplotlib.

    import matplotlib.pyplot as plt
    import inkatoms as ia

    fig, ax = plt.subplots()
    art = ia.draw(ax, atoms, view="-y", style="jmol", highlight=[12])
    ia.legend(ax, art)
"""

from . import select
from .annotate import measure
from .bonds import (ORDER_LIMITS, bond_orders, coordination, cutoff_matrix, find_bonds, find_hbonds,
                    find_rings)
from .extras import COLUMN_WIDTHS, axes_triad, colorbar, figsize, legend, legend_handles, scale_bar
from .figures import animate, inset, panels
from .fonts import use_collection_faces
from .picker import Picker, picker
from .render import StructureArtist, atom_values, draw, set_scale, structure_figure
from .slices import density_slice, read_cube, sample_plane
from .structure import Structure, as_structure, cell_from_parameters, from_smiles
from .styles import PRESETS, SCHEMES, Style, scheme_colors, style
from .view import auto_view, project, view_matrix

__version__ = "0.2.0"

__all__ = [
    # drawing
    "draw", "StructureArtist", "set_scale", "structure_figure", "atom_values",
    # structures
    "Structure", "as_structure", "from_smiles", "cell_from_parameters", "select",
    # bonds
    "find_bonds", "bond_orders", "ORDER_LIMITS", "find_hbonds", "find_rings", "coordination",
    "cutoff_matrix",
    # styles
    "Style", "style", "PRESETS", "SCHEMES", "scheme_colors",
    # annotation and figures
    "legend", "legend_handles", "scale_bar", "colorbar", "axes_triad", "measure", "figsize",
    "COLUMN_WIDTHS", "inset", "panels", "animate", "picker", "Picker",
    # volumetric data
    "density_slice", "sample_plane", "read_cube",
    # views
    "view_matrix", "auto_view", "project",
    # fonts
    "use_collection_faces",
]
