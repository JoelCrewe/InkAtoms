"""draw() and the artist it returns. The pieces themselves are made in layers.py."""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection, PathCollection
from matplotlib.colors import Colormap, Normalize, to_rgb
from matplotlib.patches import PathPatch

from . import layers as _layers
from . import select as _select
from ._elements import DATA
from .scene import SOLID, Scene, _highlight, _occlusion, atom_values
from .structure import as_structure
from .styles import mix
from .styles import style as get_style
from .view import auto_view, view_matrix

MODES = ("ballstick", "spacefill", "sticks")
AUTO_RASTERIZE = 40000   # pieces


class StructureArtist:
    """What draw() returns: the drawn pieces plus where every atom landed."""

    def __init__(self, ax, collection, structure, projected, radii, R, style, scene):
        self.ax = ax
        self.collection = collection        # None when editable=True
        self.patches = []
        self.structure = structure
        self.projected = projected          # page x, page y, depth
        self.radii = radii
        self.rotation = R
        self.style = style
        self.mappable = None
        self.values = None
        self.labels = []
        self.extras = []
        self._scene = scene

    def _rows(self, atom_id):
        rows = self.structure.index_of(atom_id)
        if len(rows) == 0:
            raise KeyError(f"no drawn atom with id {atom_id}")
        return rows

    def row(self, atom_id):
        """Row of an atom (the front-most copy if there are several)."""
        rows = self._rows(atom_id)
        return int(rows[np.argmax(self.projected[rows, 2])])

    def xy(self, atom_id):
        """Page position of an atom (the front-most copy), for annotations."""
        return tuple(self.projected[self.row(atom_id), :2])

    def xy_all(self, atom_id):
        return self.projected[self._rows(atom_id), :2]

    def radius(self, atom_id):
        return float(self.radii[self.row(atom_id)])

    def position(self, atom_id):
        """3D position (Å) of an atom (front-most copy)."""
        return self.structure.positions[self.row(atom_id)]

    def project(self, points):
        """Page (x, y) of 3D points under this drawing's view."""
        xy, _, _ = self._scene.project(np.atleast_2d(points))
        return xy

    @property
    def symbols(self):
        """Elements present, heaviest first."""
        syms = set(self.structure.symbols)
        return sorted(syms, key=lambda s: -list(DATA).index(s))

    @property
    def bounds(self):
        """(xmin, xmax, ymin, ymax) of the drawn atoms on the page."""
        x, y = self.projected[:, 0], self.projected[:, 1]
        r = self.radii
        return (float((x - r).min()), float((x + r).max()),
                float((y - r).min()), float((y + r).max()))

    def remove(self):
        if self.collection is not None:
            self.collection.remove()
        for p in self.patches + self.labels + self.extras:
            p.remove()


def draw(ax, atoms, *, view="z", up=None, rotate=None, perspective=None,
         style="jmol", mode="ballstick", depth_cue=True,
         color_by=None, cmap="viridis", vmin=None, vmax=None, norm=None,
         highlight=None, highlight_front=False, layers=None,
         bonds=True, cutoffs=None, bond_scale=1.15, bond_orders=False, pbc_bonds=None,
         bond_styles=None, contacts=None,
         vectors=None, vector_scale=1.0, vector_color="#c0392b", vector_width=0.07, vectors_front=False,
         trails=None,
         polyhedra=None, rings=None,
         labels=None, label_kw=None, constraints=None, vacancies=None,
         ghosts=None, boundary=False,
         cell=False, cell_vectors=False, planes=None,
         radii=None, depth_range=None, autoscale=True, pad=0.5, axis_off=True,
         zorder=2, rasterized="auto", editable=False, scale=None,
         **style_changes) -> StructureArtist:
    """Draw a structure on ``ax``.

    Parameters
    ----------
    atoms : Atoms, pymatgen Structure/Molecule, RDKit Mol, MDAnalysis Universe or Structure
    view : str, array or (3, 3) matrix
        Direction out of the page: ``"z"`` (top), ``"-y"`` (side), any vector, or
        ``"auto"`` for the principal axes. ``up`` and ``rotate`` (e.g. ``"20x,-30y"``)
        are passed to :func:`view_matrix`.
    perspective : float, optional
        Camera distance in Å; orthographic if None.
    style : str or Style
        Preset name (see :func:`style`); extra keyword arguments change Style fields.
    mode : {"ballstick", "spacefill", "sticks"}
    depth_cue : bool
        Fade and shrink atoms with depth.
    color_by : str or array, optional
        ``"z"``, ``"coordination"``, the name of a per-atom array or an array.
        ``cmap``, ``vmin``, ``vmax`` and ``norm`` work as in matplotlib.
    highlight : list or dict, optional
        Ids drawn unfaded with a heavier outline, or ``{id: colour}``.
        ``highlight_front`` puts them on top of everything.
    layers : dict, optional
        ``{symbol: int}``; lower layers are drawn first regardless of depth.
    bonds : bool or list
        Find bonds, draw none, or draw exactly the given ``(id_a, id_b)`` pairs.
        ``cutoffs`` overrides pair cutoffs, e.g. ``{"Mo-S": 2.85}``.
    bond_orders : bool or dict
        Guess double/triple bonds (or use RDKit's), or ``{(id_a, id_b): order}``.
    bond_styles : dict, optional
        ``{(id_a, id_b): "breaking" | "forming" | "dashed" | "hidden"}`` or a dict of
        ``color``, ``width``, ``dashed``, ``hidden``.
    contacts : "hbonds" or list, optional
        Extra dashed contacts.
    pbc_bonds : bool, optional
        Half-bonds across periodic boundaries; follows the structure by default.
    vectors : array or str, optional
        (N, 3) arrows, or the name of a per-atom array such as ``"forces"``.
    trails : dict, optional
        Motion trails, see :func:`inkatoms.layers.trails`.
    polyhedra, rings : bool or dict, optional
        Coordination polyhedra and filled rings.
    labels : str, list or dict, optional
        ``"symbol"``, ``"id"``, ``"index"``, ``"value"``, a list or ``{id: text}``;
        ``label_kw`` goes to ``ax.text``.
    constraints : bool or list, optional
        Mark fixed atoms (True uses ASE's FixAtoms).
    vacancies : (M, 3) array, optional
    ghosts : float, optional
        Show periodic images within this distance (Å) of the cell, faded.
    boundary : bool
        Repeat atoms on cell faces, edges and corners.
    cell, cell_vectors, planes : optional
        Cell outline, a/b/c arrows and lattice planes such as ``[(1, 1, 1)]``.
    radii : dict, optional
        ``{symbol: radius}`` in Å.
    depth_range : (far, near), optional
        Fix the fade range, e.g. across trajectory panels.
    scale : float, optional
        Inches per Å, see :func:`set_scale`.
    rasterized : bool or "auto"
    editable : bool
        One patch per piece, with SVG ids such as ``atom-12``.

    Returns
    -------
    StructureArtist
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    s = as_structure(atoms)
    st = get_style(style, **style_changes)
    if not depth_cue:
        st = st.replace(depth_fade=0.0, depth_shrink=0.0)

    if ghosts or boundary:
        if s.cell is None:
            raise ValueError("ghosts/boundary need a structure with a unit cell")
        s = _select.images(s, width=float(ghosts) if ghosts else 0.0)
        pbc_bonds = False
    is_image = np.asarray(s.arrays.get("_image", np.zeros(len(s), bool)), bool)
    ghost_fade = 0.6 if ghosts else 0.0
    extra_fade = np.where(is_image, ghost_fade, 0.0)
    if "_fade" in s.arrays:
        extra_fade = np.maximum(extra_fade, np.asarray(s.arrays["_fade"], float))

    if isinstance(view, str) and view == "auto":
        view = auto_view(s.positions)
    R = view_matrix(view, up=up, rotate=rotate)
    n = len(s)
    bg = np.array(to_rgb(st.background))
    scene = Scene(s.positions, R, st, perspective, depth_range, bg)

    vdw = np.array([DATA[x][1] for x in s.symbols])
    if mode == "spacefill":
        r3 = vdw.copy()
    elif mode == "sticks":
        r3 = np.full(n, st.bond_width / 2)
    else:
        r3 = st.ball_scale * vdw
    for sym, rad in (radii or {}).items():
        r3[s.symbols == sym] = rad
    if "_scale" in s.arrays:
        r3 = r3 * np.asarray(s.arrays["_scale"], float)

    if layers or "_layer" in s.arrays:
        scene.atom_layer = np.array([(layers or {}).get(sym, 0) for sym in s.symbols], float)
        if "_layer" in s.arrays:
            scene.atom_layer = scene.atom_layer + np.asarray(s.arrays["_layer"], float)
    hl, hl_colours = _highlight(s, highlight)
    xy, depth, f_atom = scene.project(s.positions) if n else (np.zeros((0, 2)), np.zeros(0), np.zeros(0))
    t_atom = np.where(hl, 1.0, scene.t(depth))
    k_atom = scene.k(t_atom)
    r_eff = r3 * k_atom * np.where(hl, st.highlight_scale, 1.0)   # bonds are clipped at this
    rr = r_eff * f_atom
    fade_atom = scene.fade(t_atom, extra_fade)

    base = np.array([st.color(x) for x in s.symbols]) if n else np.zeros((0, 3))
    values = atom_values(s, color_by, cutoffs=cutoffs, bond_scale=bond_scale)
    mappable = None
    if values is not None:
        cm = colormaps[cmap] if isinstance(cmap, str) else cmap
        if not isinstance(cm, Colormap):
            raise TypeError("cmap must be a colormap or its name")
        finite = np.isfinite(values)
        if norm is None:
            lo = np.nanmin(values) if vmin is None else vmin
            hi = np.nanmax(values) if vmax is None else vmax
            norm = Normalize(lo, hi if hi > lo else lo + 1e-9)
        base[finite] = cm(norm(values[finite]))[:, :3]
        mappable = ScalarMappable(norm=norm, cmap=cm)
        mappable.set_array(values)
    for row, colour in hl_colours.items():
        if colour is not None:
            base[row] = colour
    if st.occlusion > 0 and n:
        base = mix(base, (0, 0, 0), st.occlusion * 0.55 * _occlusion(s))

    ctx = _layers.Ctx(ax=ax, s=s, st=st, scene=scene, n=n, xy=xy, depth=depth, k_atom=k_atom,
                      r_eff=r_eff, rr=rr, fade_atom=fade_atom, extra_fade=extra_fade, hl=hl,
                      base=base, zorder=zorder)
    _layers.atoms(ctx, highlight_front=highlight_front, constraints=constraints)
    _layers.find_bond_list(ctx, bonds=bonds, mode=mode, polyhedra=polyhedra, rings=rings,
                           contacts=contacts, cutoffs=cutoffs, bond_scale=bond_scale,
                           pbc_bonds=pbc_bonds)
    _layers.polyhedra(ctx, polyhedra)   # must run before bonds()
    _layers.bonds(ctx, bond_orders=bond_orders, bond_styles=bond_styles)
    _layers.contacts(ctx, contacts)
    _layers.rings(ctx, rings)
    _layers.vectors(ctx, vectors, vector_scale, vector_color, vector_width, vectors_front)
    _layers.trails(ctx, trails)
    _layers.vacancies(ctx, vacancies)
    _layers.cell(ctx, cell=cell, cell_vectors=cell_vectors, planes=planes)
    extras = ctx.extras

    # back to front
    items = scene.items
    if rasterized == "auto":
        rasterized = len(items) > AUTO_RASTERIZE
    coll = None
    art_patches = []
    if items:
        d = np.array([it[0] for it in items])
        g = np.array([it[1] for it in items])
        o = np.array([it[2] for it in items])
        lay = np.array([it[9] for it in items])
        order = np.lexsort((o, g, np.round(d, 6), lay))
        if editable:
            for rank, q_ in enumerate(order):
                _, _, _, path, face, edge, lw, ls, gid, _ = items[q_]
                patch = PathPatch(path)
                patch.set_facecolor(face)
                patch.set_edgecolor(edge)
                patch.set_linewidth(lw)
                patch.set_linestyle(ls)
                patch.set_zorder(zorder + 0.999 * rank / len(order))
                patch.set_gid(gid)
                patch.set_rasterized(rasterized)
                ax.add_patch(patch)
                art_patches.append(patch)
        else:
            coll = PathCollection([items[q_][3] for q_ in order], zorder=zorder)
            coll.set_facecolor([items[q_][4] for q_ in order])
            coll.set_edgecolor([items[q_][5] for q_ in order])
            coll.set_linewidth([items[q_][6] for q_ in order])
            styles = [items[q_][7] for q_ in order]
            if any(ls != SOLID for ls in styles):
                coll.set_linestyle(styles)
            coll.set_joinstyle("round")
            coll.set_rasterized(rasterized)
            coll.set_gid("inkatoms-structure")
            ax.add_collection(coll)
    else:
        coll = PatchCollection([], zorder=zorder)
        ax.add_collection(coll)

    P = np.column_stack([xy, depth]) if n else np.zeros((0, 3))
    art = StructureArtist(ax, coll, s, P, rr, R, st, scene)
    art.patches = art_patches
    art.mappable = mappable
    art.values = values
    art.extras = extras

    if autoscale and n:
        x0, x1, y0, y1 = art.bounds
        if scene.extent:
            c = np.array(scene.extent)
            x0, x1 = min(x0, c[:, 0].min()), max(x1, c[:, 0].max())
            y0, y1 = min(y0, c[:, 1].min()), max(y1, c[:, 1].max())
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    if axis_off:
        ax.set_axis_off()
    if scale is not None:
        set_scale(ax, scale)
    if labels is not None:
        from .annotate import _draw_labels
        art.labels = _draw_labels(art, labels, values, base, fade_atom, dict(label_kw or {}), zorder)
    return art


def set_scale(ax, inches_per_angstrom, centre=None):
    """Set the axes limits so 1 Å is exactly inches_per_angstrom inches on the page,
    centred on the current view (or on centre=(x, y) in Å). Use the same value
    in every figure to keep structures at one scale across a paper."""
    fig = ax.figure
    box = ax.get_position(original=True)
    w_in = box.width * fig.get_figwidth()
    h_in = box.height * fig.get_figheight()
    if centre is None:
        centre = (np.mean(ax.get_xlim()), np.mean(ax.get_ylim()))
    hx = 0.5 * w_in / inches_per_angstrom
    hy = 0.5 * h_in / inches_per_angstrom
    ax.set_xlim(centre[0] - hx, centre[0] + hx)
    ax.set_ylim(centre[1] - hy, centre[1] + hy)
    return ax


def structure_figure(atoms, inches_per_angstrom=0.3, *, pad=0.4, max_size=12.0, **draw_kwargs):
    """A new figure sized to the structure at a fixed scale (inches per Å),
    with the structure filling it. Returns (fig, ax, artist)."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    art = draw(ax, atoms, pad=0.0, **draw_kwargs)
    x0, x1, y0, y1 = art.bounds
    w = (x1 - x0 + 2 * pad) * inches_per_angstrom
    h = (y1 - y0 + 2 * pad) * inches_per_angstrom
    if max(w, h) > max_size:
        plt.close(fig)
        raise ValueError(f"figure would be {w:.1f} x {h:.1f} in; lower inches_per_angstrom "
                         f"or raise max_size")
    fig.set_size_inches(w, h)
    set_scale(ax, inches_per_angstrom, centre=(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
    return fig, ax, art
