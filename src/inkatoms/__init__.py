"""Vector drawings of atomic structures in matplotlib."""

from . import layers, scene, select
from .annotate import measure
from .bonds import (ORDER_LIMITS, bond_orders, coordination, cutoff_matrix, find_bonds, find_hbonds,
                    find_rings)
from .compare import Comparison, compare, draw_comparison, kabsch, match
from .events import (BondEvent, Reaction, active_colour, event_strip, find_events, find_reactions,
                     group_reactions, neighbour_counts, plot_neighbours)
from .extras import COLUMN_WIDTHS, axes_triad, colorbar, figsize, legend, legend_handles, scale_bar
from .figures import animate, inset, panels
from .fonts import use_collection_faces
from .picker import Picker, picker
from .render import StructureArtist, atom_values, draw, set_scale, structure_figure
from .slices import density_slice, read_cube, sample_plane
from .structure import Structure, as_structure, cell_from_parameters, from_smiles
from .styles import PRESETS, SCHEMES, Style, scheme_colors, style
from .tracking import LinkModel, LinkModels, fit_link_model, link, link_accuracy
from .trajectory import as_frames, mobile, track, unwrap
from .view import auto_view, project, view_matrix

__version__ = "0.3.0.dev0"

__all__ = [
    "BondEvent", "COLUMN_WIDTHS", "Comparison", "LinkModel", "LinkModels", "ORDER_LIMITS", "PRESETS",
    "Picker", "Reaction", "SCHEMES", "Structure", "StructureArtist", "Style", "active_colour", "animate",
    "as_frames", "as_structure", "atom_values", "auto_view", "axes_triad", "bond_orders",
    "cell_from_parameters", "colorbar", "compare", "coordination", "cutoff_matrix", "density_slice",
    "draw", "draw_comparison", "event_strip", "figsize", "find_bonds", "find_events", "find_hbonds",
    "find_reactions", "find_rings", "fit_link_model", "from_smiles", "group_reactions", "inset", "kabsch",
    "legend", "legend_handles", "link", "link_accuracy", "match", "measure", "mobile", "neighbour_counts",
    "panels", "picker", "plot_neighbours", "project", "read_cube", "sample_plane", "scale_bar",
    "scheme_colors", "select", "set_scale", "structure_figure", "style", "track", "unwrap",
    "use_collection_faces", "view_matrix",
]
