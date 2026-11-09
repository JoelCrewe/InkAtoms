"""Draw a structure onto a matplotlib Axes.

draw() prepares the view, radii, depth cueing and colours, then calls the
layer functions in layers.py, each adding pieces (atoms, bonds, polyhedra,
...) to one depth-sorted Scene, and finally assembles the scene back to front
into a single vector artist.
"""

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
AUTO_RASTERIZE = 40000   # rasterize="auto" switches on above this many drawn pieces


class StructureArtist:
    """What draw() returns: the drawn pieces plus where every atom landed."""

    def __init__(self, ax, collection, structure, projected, radii, R, style, scene):
        self.ax = ax
        self.collection = collection        # PatchCollection (None when editable=True)
        self.patches = []                   # individual patches when editable=True
        self.structure = structure          # what was drawn (after images/ghosts)
        self.projected = projected          # (N, 3) page x, page y, depth
        self.radii = radii                  # (N,) drawn radius on the page, Å
        self.rotation = R                   # rows: page right, up, out in structure coordinates
        self.style = style
        self.mappable = None                # ScalarMappable for colorbar() when color_by= was used
        self.values = None                  # the per-atom values behind color_by
        self.labels = []                    # Text artists from labels=
        self.extras = []                    # other artists added by draw (cell-vector labels, ...)
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
    """Draw a structure on ax and return a StructureArtist.

    Structure and view
      atoms        ASE Atoms, pymatgen Structure/Molecule, RDKit Mol, MDAnalysis
                   Universe, (symbols, positions) or Structure
      view         direction out of the page: "z" (top), "-y" (side), ..., a vector,
                   a 3x3 matrix, or "auto" (principal axes, widest face)
      up, rotate   see view_matrix; rotate="20x,-30y" tilts about the page axes
      perspective  camera distance (Å) from the structure centre; None is orthographic

    Look
      style        jmol | cpk | vesta | colourblind | greyscale | bw | dark, or a Style;
                   extra keyword arguments change style fields (shading="soft", ...)
      mode         ballstick | spacefill | sticks
      depth_cue    False draws every atom at full colour and size (no fade, no
                   shrink); the style's depth_fade / depth_shrink tune it when on
      color_by     colour atoms by a value: "z", "coordination", an array name
                   ("charges", "forces", ...) or an array; cmap, vmin, vmax, norm as
                   in matplotlib; then inkatoms.colorbar(artist)
      highlight    ids drawn unfaded with a heavier outline, or {id: colour};
                   highlight_front=True keeps them on top of everything
      layers       {symbol: int}: draw lower layers first whatever their depth (atoms
                   within a layer still sort and fade by depth), e.g. {"S": 0, "Mo": 1,
                   "H": 2} keeps a Mo network and adsorbed H visible in a dense slice

    Bonds
      bonds, cutoffs, bond_scale   bonds=True finds bonds (cutoffs={"Mo-S": 2.85}
                   overrides pairs, in Å), False draws none, or give an explicit list
                   of bonded id pairs [(id_a, id_b), ...] to draw exactly those
      bond_orders  True guesses double/triple bonds (or uses known RDKit orders),
                   or {(id_a, id_b): order}; order 1.5 is drawn as solid + dashed
      bond_styles  {(id_a, id_b): "breaking" | "forming" | "dashed" | "hidden" or
                   dict(color=, width=, dashed=, hidden=)}
      contacts     extra dashed contacts: "hbonds" or [(id_a, id_b), ...]
      pbc_bonds    draw half-bonds across periodic boundaries (default: follow pbc)

    Extras
      vectors      per-atom arrows: array (N, 3) or an array name ("forces"),
                   scaled by vector_scale; vector_color ("atom" for element colour);
                   vectors_front=True draws them over everything
      trails       motion trails from a trajectory: dict(frames=, ids=[...] | "mobile",
                   n=, smooth=, color="atom" | "time" | colour, width=, fade=)
      polyhedra    True (metal-centred) or dict(centers=, ligands=, alpha=, color=,
                   edgecolor=, hide_bonds=, min_ligands=)
      rings        True or dict(color=, alpha=, aromatic="auto"|True|False, max_size=)
      labels       "symbol" | "id" | "index" | "value" | list | {id: text}; label_kw
                   passed to ax.text (fontsize, color="auto", offset=(dx, dy), ...)
      constraints  True (ASE FixAtoms) or ids: a cross on fixed atoms
      vacancies    (M, 3) positions drawn as dashed empty circles
      ghosts       width (Å): periodic images outside the cell drawn faded
      boundary     repeat atoms on cell faces/edges/corners (as VESTA does)
      cell, cell_vectors, planes   cell outline; labelled a/b/c arrows; lattice
                   planes [(1, 1, 1), {"hkl": (0, 0, 1), "offset": 0.5, "color": ...}]

    Output
      radii        {symbol: radius in Å}; depth_range=(far, near) fixes the fade
      scale        inches per Å (see set_scale)
      rasterized   True / False / "auto" (rasterize very large drawings)
      editable     one patch per piece with gids like "atom-12" (for Illustrator)
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    s = as_structure(atoms)
    st = get_style(style, **style_changes)
    if not depth_cue:
        st = st.replace(depth_fade=0.0, depth_shrink=0.0)

    # ---- periodic images
    if ghosts or boundary:
        if s.cell is None:
            raise ValueError("ghosts/boundary need a structure with a unit cell")
        s = _select.images(s, width=float(ghosts) if ghosts else 0.0)
        pbc_bonds = False
    is_image = np.asarray(s.arrays.get("_image", np.zeros(len(s), bool)), bool)
    ghost_fade = 0.6 if ghosts else 0.0
    extra_fade = np.where(is_image, ghost_fade, 0.0)
    if "_fade" in s.arrays:              # per-atom extra fade, e.g. the ghosts of a comparison
        extra_fade = np.maximum(extra_fade, np.asarray(s.arrays["_fade"], float))

    if isinstance(view, str) and view == "auto":
        view = auto_view(s.positions)
    R = view_matrix(view, up=up, rotate=rotate)
    n = len(s)
    bg = np.array(to_rgb(st.background))
    scene = Scene(s.positions, R, st, perspective, depth_range, bg)

    # ---- radii (Å, before depth shrink)
    vdw = np.array([DATA[x][1] for x in s.symbols])
    if mode == "spacefill":
        r3 = vdw.copy()
    elif mode == "sticks":
        r3 = np.full(n, st.bond_width / 2)
    else:
        r3 = st.ball_scale * vdw
    for sym, rad in (radii or {}).items():
        r3[s.symbols == sym] = rad
    if "_scale" in s.arrays:             # per-atom radius factor (e.g. comparison ghosts)
        r3 = r3 * np.asarray(s.arrays["_scale"], float)

    if layers or "_layer" in s.arrays:
        scene.atom_layer = np.array([(layers or {}).get(sym, 0) for sym in s.symbols], float)
        if "_layer" in s.arrays:         # per-atom layer offsets
            scene.atom_layer = scene.atom_layer + np.asarray(s.arrays["_layer"], float)
    hl, hl_colours = _highlight(s, highlight)
    xy, depth, f_atom = scene.project(s.positions) if n else (np.zeros((0, 2)), np.zeros(0), np.zeros(0))
    t_atom = np.where(hl, 1.0, scene.t(depth))
    k_atom = scene.k(t_atom)
    r_eff = r3 * k_atom * np.where(hl, st.highlight_scale, 1.0)   # 3D radius used for clipping
    rr = r_eff * f_atom                                           # drawn radius on the page
    fade_atom = scene.fade(t_atom, extra_fade)

    # ---- colours
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
    _layers.polyhedra(ctx, polyhedra)          # before bonds: it hides centre-ligand bonds
    _layers.bonds(ctx, bond_orders=bond_orders, bond_styles=bond_styles)
    _layers.contacts(ctx, contacts)
    _layers.rings(ctx, rings)
    _layers.vectors(ctx, vectors, vector_scale, vector_color, vector_width, vectors_front)
    _layers.trails(ctx, trails)
    _layers.vacancies(ctx, vacancies)
    _layers.cell(ctx, cell=cell, cell_vectors=cell_vectors, planes=planes)
    extras = ctx.extras

    # ---- assemble, back to front
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
