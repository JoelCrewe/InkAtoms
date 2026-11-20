"""One function per kind of piece (atoms, bonds, polyhedra, ...). draw() calls them in order."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.colors import to_rgb

from ._elements import DATA
from .bonds import NONMETALS, find_bonds, find_hbonds, find_rings
from .scene import (DASHED, _MULTI, _NAMED_BOND_STYLES, _arrow, _bond_polygon, _cell_edges, _circle,
                    _dashes, _hull_faces, _plane_polygon, _poly, _resolve_orders, _wedge)
from .styles import edge_of, mix

_EMPTY = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3), int))

# "matte" shading: a diffuse sphere, brightness ambient + diffuse x cos(normal, view),
# no highlight. The defaults match OVITO's Tachyon renders: about 1.1 x the base colour
# at the centre, darkening to about 0.85 x at the rim.
_MATTE_AMBIENT, _MATTE_DIFFUSE = 0.85, 0.25
_MATTE_STEPS = 20
_MATTE_OFFSET = 0.12


def _matte(face, cos):
    return np.clip(np.asarray(face, float) * (_MATTE_AMBIENT + _MATTE_DIFFUSE * cos), 0.0, 1.0)


@dataclass
class Ctx:
    """State shared by the layers for one drawing."""

    ax: object
    s: object
    st: object
    scene: object
    n: int
    xy: np.ndarray
    depth: np.ndarray
    k_atom: np.ndarray        # depth shrink factor
    r_eff: np.ndarray         # 3D radius, bonds are clipped at this
    rr: np.ndarray            # radius on the page
    fade_atom: np.ndarray
    extra_fade: np.ndarray
    hl: np.ndarray
    base: np.ndarray          # colours before fading
    zorder: float
    bond_list: tuple = _EMPTY
    T: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    hidden_pairs: set = field(default_factory=set)
    draw_bonds: bool = False
    extras: list = field(default_factory=list)


def atoms(ctx, highlight_front=False, constraints=None):
    """Atoms with their shading, partial-occupancy pies and constraint crosses."""
    s, st, scene, n, xy = ctx.s, ctx.st, ctx.scene, ctx.n, ctx.xy
    depth, k_atom, rr, fade_atom, hl = ctx.depth, ctx.k_atom, ctx.rr, ctx.fade_atom, ctx.hl
    base = ctx.base
    fixed = np.zeros(n, bool)
    if constraints is True:
        fixed = np.asarray(s.arrays.get("fixed", fixed), bool)
    elif constraints is not None:
        fixed = np.isin(s.ids, np.atleast_1d(constraints))
    occ = s.occupancy or (None,) * n

    for a in range(n):
        x, y = xy[a]
        r = rr[a]
        fade = fade_atom[a]
        face = base[a]
        edge_rule = st.edge_colors.get(s.symbols[a], st.atom_edgecolor)
        if hl[a] and st.highlight_edgecolor is not None:
            edge_rule = st.highlight_edgecolor
        edge = edge_of(face, edge_rule, st.darken)
        lw = (st.highlight_edgewidth if hl[a] else st.atom_edgewidth) * k_atom[a]
        if st.shading == "soft":
            body = mix(face, (0, 0, 0), 0.12)
        elif st.shading == "matte":
            body = _matte(face, 0.5 / _MATTE_STEPS)
        else:
            body = face
        da = depth[a] + (1e6 if (highlight_front and hl[a]) else 0.0)
        gid = f"atom-{s.ids[a]}"
        scene.add(da, a, 0, _circle((x, y), r), scene.rgba(body, fade), scene.rgba(edge, fade), lw, gid=gid)
        pie = occ[a]
        if pie:
            theta = 90.0
            for sym, frac in pie:
                c = to_rgb(st.colors.get(sym) or st.color(sym))
                scene.add(da, a, 1, _wedge((x, y), r * 0.985, theta - 360 * frac, theta),
                          scene.rgba(c, fade), (0, 0, 0, 0), 0.0, gid=gid)
                theta -= 360 * frac
            if theta > 90.0 - 359.9:   # vacancy share
                scene.add(da, a, 1, _wedge((x, y), r * 0.985, 90.0 - 360.0, theta),
                          scene.rgba((1, 1, 1), fade), (0, 0, 0, 0), 0.0, gid=gid)
            scene.add(da, a, 40, _circle((x, y), r), (0, 0, 0, 0), scene.rgba(edge, fade), lw, gid=gid)
        elif st.shading == "soft":
            steps = 6
            for k in range(1, steps):
                fk = k / steps
                rad = r * (1 - 0.82 * fk)
                off = (r - rad) * 0.62
                c = mix(face, (1, 1, 1), 0.55 * fk ** 1.3) if fk > 0.3 else face
                scene.add(da, a, k, _circle((x - off * 0.7071, y + off * 0.7071), rad),
                          scene.rgba(c, fade), (0, 0, 0, 0), 0.0, gid=gid)
        elif st.shading == "matte":
            # nested discs, equal steps in brightness, lit slightly from the upper left
            for k in range(1, _MATTE_STEPS):
                cos = k / _MATTE_STEPS
                rad = r * np.sqrt(1.0 - cos * cos)
                off = (r - rad) * _MATTE_OFFSET
                scene.add(da, a, k, _circle((x - off * 0.7071, y + off * 0.7071), rad),
                          scene.rgba(_matte(face, cos + 0.5 / _MATTE_STEPS), fade), (0, 0, 0, 0), 0.0,
                          gid=gid)
        if st.shading == "glossy":
            for k, (offf, radf, alpha) in enumerate(((0.30, 0.30, 0.30), (0.36, 0.13, 0.75)), start=20):
                scene.add(da, a, k, _circle((x - offf * r, y + offf * r), radf * r),
                          (1, 1, 1, alpha * (1 - fade)), (0, 0, 0, 0), 0.0, gid=gid)
        elif st.shading not in ("flat", "soft", "matte"):
            raise ValueError("shading must be flat, glossy, soft or matte")
        if fixed[a]:
            for ang in (45.0, -45.0):
                u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]) * 0.62 * r
                scene.add(da, a, 70, _poly(_bond_polygon(np.array([x, y]) - u, np.array([x, y]) + u, 0.07 * r)), scene.rgba((0.1, 0.1, 0.1), fade, 0.85),
                          (0, 0, 0, 0), 0.0, gid=gid)


def find_bond_list(ctx, bonds=True, mode="ballstick", polyhedra=None, rings=None, contacts=None,
                   cutoffs=None, bond_scale=1.15, pbc_bonds=None):
    """Work out which bonds exist (found, or an explicit id list) for the bond-based layers."""
    s, n = ctx.s, ctx.n
    explicit = not isinstance(bonds, (bool, np.bool_)) and bonds is not None
    draw_bonds = (explicit or bool(bonds)) and mode != "spacefill"
    need_bonds = draw_bonds or polyhedra or rings or contacts == "hbonds"
    if explicit:
        # explicit (id_a, id_b) pairs, no periodic shifts
        pairs = np.asarray(list(bonds), int).reshape(-1, 2)
        first = {int(i): r for r, i in reversed(list(enumerate(s.ids)))}
        missing = [int(x) for x in pairs.ravel() if int(x) not in first]
        if missing:
            raise KeyError(f"bonds refer to ids not in the structure: {sorted(set(missing))[:8]}")
        bi = np.array([first[int(a)] for a in pairs[:, 0]], int)
        bj = np.array([first[int(b)] for b in pairs[:, 1]], int)
        bond_list = (bi, bj, np.zeros((len(bi), 3), int))
    elif need_bonds and n > 1:
        bond_list = find_bonds(s, cutoffs=cutoffs, scale=bond_scale, pbc=pbc_bonds)
    else:
        bond_list = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3), int))
    bi, bj, shifts = bond_list
    T = shifts @ s.cell if (s.cell is not None and len(shifts)) else np.zeros((len(bi), 3))
    ctx.bond_list, ctx.T, ctx.draw_bonds = bond_list, T, draw_bonds


def polyhedra(ctx, polyhedra=None):
    """Coordination polyhedra; records the centre-ligand bonds they hide."""
    s, st, scene, n, extra_fade = ctx.s, ctx.st, ctx.scene, ctx.n, ctx.extra_fade
    base, T, bond_list = ctx.base, ctx.T, ctx.bond_list
    bi, bj, shifts = bond_list
    hidden_pairs = set()
    if polyhedra:
        spec = {} if polyhedra is True else dict(polyhedra)
        centres = spec.get("centers", spec.get("centres"))
        if centres is None:
            centres = sorted({x for x in s.symbols if x not in NONMETALS})
        centres = set(np.atleast_1d(centres))
        ligands = spec.get("ligands")
        ligands = None if ligands is None else set(np.atleast_1d(ligands))
        alpha = spec.get("alpha", 0.38)
        min_lig = spec.get("min_ligands", 4)
        neigh = [[] for _ in range(n)]
        for i, j, tvec in zip(bi, bj, T):
            neigh[i].append((j, s.positions[j] + tvec))
            neigh[j].append((i, s.positions[i] - tvec))
        for c in range(n):
            if s.symbols[c] not in centres and s.ids[c] not in centres:
                continue
            if ligands is not None:
                ligs = [(r_, p) for r_, p in neigh[c] if s.symbols[r_] in ligands]
            elif s.symbols[c] not in NONMETALS:
                ligs = [(r_, p) for r_, p in neigh[c] if s.symbols[r_] in NONMETALS]
            else:
                ligs = list(neigh[c])
            if len(ligs) < min_lig:
                continue
            pts = np.array([p for _, p in ligs])
            colour = np.array(to_rgb(spec["color"])) if spec.get("color") else base[c]
            edge_c = edge_of(colour, spec.get("edgecolor", "darker"), st.darken)
            for face_idx in _hull_faces(pts):
                fxy, fd, _ = scene.project(pts[face_idx])
                t = float(scene.t(fd.mean()))
                fade = float(scene.fade(t, extra_fade[c]))
                scene.add(fd.mean(), -1, 0, _poly(fxy), scene.rgba(colour, fade, alpha),
                          scene.rgba(edge_c, fade, min(1.0, alpha + 0.45)), spec.get("edgewidth", 0.5),
                          gid=f"polyhedron-{s.ids[c]}")
            if spec.get("hide_bonds", True):
                hidden_pairs.update(frozenset((c, r_)) for r_, _ in ligs)
    ctx.hidden_pairs = hidden_pairs


def bonds(ctx, bond_orders=False, bond_styles=None):
    """Bonds as clipped, depth-sorted (half-)bond polygons, with orders and styles."""
    s, st, scene, r_eff, extra_fade = ctx.s, ctx.st, ctx.scene, ctx.r_eff, ctx.extra_fade
    hl, base, bond_list, T, hidden_pairs = ctx.hl, ctx.base, ctx.bond_list, ctx.T, ctx.hidden_pairs
    draw_bonds = ctx.draw_bonds
    bi, bj, shifts = bond_list
    if draw_bonds and len(bi):
        orders = _resolve_orders(s, bond_orders, bond_list)
        styles_by_ids = {}
        for key, value in (bond_styles or {}).items():
            value = _NAMED_BOND_STYLES[value] if isinstance(value, str) else dict(value)
            styles_by_ids[frozenset(key)] = value
        split = isinstance(st.bond_color, str) and st.bond_color == "split"
        bond_rgb = None if split else np.array(to_rgb(st.bond_color))
        for i, j, tvec, order in zip(bi, bj, T, orders):
            if frozenset((i, j)) in hidden_pairs:
                continue
            bst = styles_by_ids.get(frozenset((int(s.ids[i]), int(s.ids[j]))), {})
            if bst.get("hidden"):
                continue
            dashed_all = bool(bst.get("dashed"))
            if dashed_all:
                order = 1
            w = st.bond_width * bst.get("width", 1.0)
            own_rgb = np.array(to_rgb(bst["color"])) if bst.get("color") else None
            pi, pj = s.positions[i], s.positions[j] + tvec
            v = pj - pi
            L = np.linalg.norm(v)
            u = v / L
            pxy, _, _ = scene.project(np.array([pi, pj]))
            d2 = pxy[1] - pxy[0]
            n2 = np.hypot(*d2)
            perp = np.array([-d2[1], d2[0]]) / n2 if n2 > 1e-9 else np.zeros(2)
            if order > 1 and n2 < 0.25 * L:
                order = 1   # bond points at the viewer; parallel lines would overlap
            for off, width, dashed in _MULTI.get(order, _MULTI[min(int(round(order)), 3)]):
                off, width = off * w, width * w
                dashed = dashed or dashed_all
                reach = abs(off) + width / 2
                si = np.sqrt(max(r_eff[i] ** 2 - reach ** 2, 0.0))
                sj = np.sqrt(max(r_eff[j] ** 2 - reach ** 2, 0.0))
                if L <= si + sj:
                    continue  # hidden inside the atoms
                ai, aj = pi + u * si, pj - u * sj
                mid = 0.5 * (ai + aj)
                if split or tvec.any():
                    halves = [(ai, mid, i), (aj - tvec, mid - tvec, j)]
                else:
                    halves = [(ai, aj, None)]
                for start, end, owner in halves:
                    pieces = _dashes(start, end, st.dash_length, st.dash_gap) if dashed else [(start, end)]
                    for p0, p1 in pieces:
                        q, d, f = scene.project(np.array([p0, p1]))
                        dmid = float(d.mean())
                        if owner is None:
                            t = 1.0 if (hl[i] and hl[j]) else float(scene.t(dmid))
                            xf = max(extra_fade[i], extra_fade[j])
                        else:
                            t = 1.0 if hl[owner] else float(scene.t(dmid))
                            xf = extra_fade[owner]
                        k = float(scene.k(t)) * float(f.mean())
                        shift2 = perp * off * k
                        poly = _bond_polygon(q[0] + shift2, q[1] + shift2, 0.5 * width * k)
                        if poly is None:
                            continue
                        if own_rgb is not None:
                            face = own_rgb
                        elif split and owner is not None:
                            face = base[owner]
                        else:
                            face = bond_rgb if bond_rgb is not None else base[i]
                        edge = edge_of(face, st.bond_edgecolor, st.darken)
                        fade = float(scene.fade(t, xf))
                        scene.add(dmid, -1, 0, _poly(poly), scene.rgba(face, fade),
                                  scene.rgba(edge, fade), st.bond_edgewidth * k,
                                  gid=f"bond-{s.ids[i]}-{s.ids[j]}",
                                  layer=min(scene.atom_layer[i], scene.atom_layer[j])
                                  if scene.atom_layer is not None else 0)


def contacts(ctx, contacts=None):
    """Dashed contacts: hydrogen bonds or given id pairs."""
    s, st, scene, r_eff, bond_list = ctx.s, ctx.st, ctx.scene, ctx.r_eff, ctx.bond_list
    if contacts is not None:
        if isinstance(contacts, str):
            if contacts != "hbonds":
                raise ValueError("contacts must be 'hbonds' or a list of id pairs")
            pairs = find_hbonds(s, bond_list)
        else:
            pairs = [(s.index_of(a)[0], s.index_of(b)[0]) for a, b in contacts]
        crgb = np.array(to_rgb(st.contact_color))
        for i, j in pairs:
            pi, pj = s.positions[i], s.positions[j]
            v = pj - pi
            L = np.linalg.norm(v)
            if L <= r_eff[i] + r_eff[j]:
                continue
            u = v / L
            for p0, p1 in _dashes(pi + u * r_eff[i], pj - u * r_eff[j], st.dash_length, st.dash_gap):
                q, d, f = scene.project(np.array([p0, p1]))
                t = float(scene.t(d.mean()))
                k = float(scene.k(t)) * float(f.mean())
                poly = _bond_polygon(q[0], q[1], 0.5 * st.contact_width * k)
                if poly is not None:
                    scene.add(d.mean(), -1, 0, _poly(poly), scene.rgba(crgb, float(scene.fade(t))),
                              (0, 0, 0, 0), 0.0, gid=f"contact-{s.ids[i]}-{s.ids[j]}")


def rings(ctx, rings=None):
    """Filled rings, with circles in aromatic ones."""
    s, scene, bond_list = ctx.s, ctx.scene, ctx.bond_list
    if rings:
        spec = {} if rings is True else dict(rings)
        ring_rgb = np.array(to_rgb(spec.get("color", "#8c8c8c")))
        alpha = spec.get("alpha", 0.28)
        aromatic = spec.get("aromatic", False)
        for ring in find_rings(s, bond_list, max_size=spec.get("max_size", 8)):
            ring = np.asarray(ring)
            rxy, rd, _ = scene.project(s.positions[ring])
            dm = float(rd.mean())
            t = float(scene.t(dm))
            fade = float(scene.fade(t))
            scene.add(dm, -1, 0, _poly(rxy), scene.rgba(ring_rgb, fade, alpha),
                      (0, 0, 0, 0), 0.0, gid="ring")
            lengths = np.linalg.norm(s.positions[ring] - s.positions[np.roll(ring, 1)], axis=1)
            is_arom = aromatic is True or (aromatic == "auto" and len(ring) in (5, 6)
                                           and np.ptp(lengths) < 0.05)
            if is_arom:
                c = rxy.mean(axis=0)
                rad = 0.55 * np.linalg.norm(rxy - c, axis=1).mean()
                scene.add(dm + 1e-3, -1, 1, _circle(tuple(c), rad), (0, 0, 0, 0),
                          scene.rgba(edge_of(ring_rgb, "darker", 0.35), fade), 1.0, gid="ring")


def vectors(ctx, vectors=None, vector_scale=1.0, vector_color="#c0392b", vector_width=0.07, front=False):
    """Per-atom arrows."""
    s, scene, n, fade_atom, base = ctx.s, ctx.scene, ctx.n, ctx.fade_atom, ctx.base
    if vectors is not None:
        vec = np.asarray(s.arrays[vectors] if isinstance(vectors, str) else vectors, float)
        if vec.shape != (n, 3):
            raise ValueError(f"vectors must be (N, 3) for {n} atoms, got {vec.shape}")
        for a in range(n):
            vv = vec[a] * vector_scale
            if not np.any(vv):
                continue
            colour = base[a] if (isinstance(vector_color, str) and vector_color == "atom") \
                else np.array(to_rgb(vector_color))
            _arrow(scene, s.positions[a], s.positions[a] + vv, vector_width, colour, fade_atom[a],
                   a, 60, gid=f"vector-{s.ids[a]}", layer=1e9 if front else None)


def vacancies(ctx, vacancies=None):
    """Dashed empty circles at vacant sites."""
    st, scene = ctx.st, ctx.scene
    if vacancies is not None:
        vac = np.atleast_2d(np.asarray(vacancies, float))
        vxy, vd, vf = scene.project(vac)
        for p, d, f in zip(vxy, vd, vf):
            t = float(scene.t(d))
            rad = st.vacancy_radius * float(scene.k(t)) * f
            # always on top, otherwise a vacancy under an atom disappears
            scene.add(d + 1e6, -1, 5, _circle(tuple(p), rad), (0, 0, 0, 0),
                      scene.rgba((0.3, 0.3, 0.3), float(scene.fade(t))), 0.9, ls=DASHED, gid="vacancy")
            scene.extent.extend([p - rad, p + rad])


def cell(ctx, cell=False, cell_vectors=False, planes=None):
    """Cell outline, labelled lattice vectors and lattice planes."""
    ax, s, st, scene, zorder = ctx.ax, ctx.s, ctx.st, ctx.scene, ctx.zorder
    extras = []
    if (cell or cell_vectors or planes) and s.cell is None:
        raise ValueError("cell, cell_vectors and planes need a structure with a unit cell")
    if cell:
        cr = np.array(to_rgb(st.cell_color))
        nseg = 12
        for p0, vec in _cell_edges(s.origin, *s.cell):
            for q_ in range(nseg):
                pq, dq, fq = scene.project(np.array([p0 + vec * q_ / nseg, p0 + vec * (q_ + 1) / nseg]))
                dmid = float(dq.mean())
                t = float(scene.t(dmid))
                k = float(scene.k(t)) * float(fq.mean())
                ext = (pq[1] - pq[0]) * 0.02
                poly = _bond_polygon(pq[0] - ext, pq[1] + ext, 0.5 * st.cell_width * k)
                if poly is not None:
                    scene.add(dmid, -2, 0, _poly(poly), scene.rgba(cr, float(scene.fade(t))),
                              (0, 0, 0, 0), 0.0, gid="cell")
            scene.extent.extend(scene.project(np.array([p0, p0 + vec]))[0])
    if cell_vectors:
        frac = 1.0 if cell_vectors is True else float(cell_vectors)
        for vec, name, colour in zip(s.cell, "abc", ("#c0392b", "#27ae60", "#2471a3")):
            tip = s.origin + vec * frac
            _arrow(scene, s.origin, tip, 1.4 * st.cell_width, np.array(to_rgb(colour)), 0.0, -3, 0, gid=f"cell-{name}")
            (txy,), _, _ = scene.project(tip[None, :] + 0.08 * vec[None, :] / max(np.linalg.norm(vec), 1e-9) * 3)
            extras.append(ax.text(*txy, name, color=colour, fontsize=9, fontstyle="italic",
                                  ha="center", va="center", zorder=zorder + 2))
    for spec in (planes or []):
        spec = {"hkl": spec} if not isinstance(spec, dict) else dict(spec)
        poly3 = _plane_polygon(s.cell, s.origin, spec["hkl"], spec.get("offset", 1.0))
        if poly3 is None:
            continue
        pxy, pd, _ = scene.project(poly3)
        colour = np.array(to_rgb(spec.get("color", "#8e44ad")))
        alpha = spec.get("alpha", 0.3)
        t = float(scene.t(pd.mean()))
        scene.add(pd.mean(), -1, 0, _poly(pxy), scene.rgba(colour, float(scene.fade(t)), alpha),
                  (*colour, min(1.0, alpha + 0.4)), 0.8, gid=f"plane-{''.join(map(str, spec['hkl']))}")
        scene.extent.extend(pxy)
    ctx.extras = extras


def trails(ctx, trails=None):
    """Motion trails, from ``draw(..., trails=dict(...))``.

    Parameters
    ----------
    trails : dict
        With the keys below.
    frames : list
        Trajectory frames, in the same coordinates as the drawing.
    ids : list or "mobile"
        Atoms to follow; "mobile" picks the ``n`` (default 8) that move most.
    style : {"arrow", "ghosts", "line"}
        Smoothed dashed route with an arrowhead, faded copies at earlier positions
        (``count``), or every step.
    color : "atom", "time" or a colour
    width : float
        Line width in Å (default 0.1).
    smooth, points, cmap, fade, id_key
        Smoothing window for "line", points along the "arrow" route, colormap for
        ``color="time"``, fading of old "line" segments, and the per-atom id array.
    """
    if not trails:
        return
    from matplotlib import colormaps

    from .trajectory import as_frames, mobile, smooth, track, unwrap

    s, st, scene, base = ctx.s, ctx.st, ctx.scene, ctx.base
    spec = dict(trails)
    structs, _ = as_frames(spec["frames"], ids=spec.get("id_key", "uid"))
    ids = spec.get("ids", "mobile")
    if isinstance(ids, str) and ids == "mobile":
        ids = mobile(structs, n=spec.get("n", 8), symbols=spec.get("symbols"), among=s.ids)
    ids = [int(i) for i in np.atleast_1d(ids)]
    paths = track(structs, ids)
    cell0 = structs[0].cell
    if cell0 is not None and structs[0].pbc.any():
        paths = unwrap(paths, cell0, structs[0].pbc)
    paths = np.stack([smooth(paths[:, k], spec.get("smooth", 3)) for k in range(len(ids))], axis=1)
    colour_mode = spec.get("color", "atom")
    cmap = colormaps[spec.get("cmap", "viridis")]
    width, fade_old = spec.get("width", 0.1), spec.get("fade", 0.85)
    row_of = {int(i): r for r, i in enumerate(s.ids)}
    sym_of = {}
    for fr in structs:
        sym_of.update(zip(map(int, fr.ids), fr.symbols))
    style_name = spec.get("style", "arrow")
    if style_name not in ("arrow", "ghosts", "line"):
        raise ValueError("trails style must be 'arrow', 'ghosts' or 'line'")
    for k, atom_id in enumerate(ids):
        if isinstance(colour_mode, str) and colour_mode == "atom":
            fixed = base[row_of[atom_id]] if atom_id in row_of else np.array(st.color(sym_of[atom_id]))
        elif isinstance(colour_mode, str) and colour_mode == "time":
            fixed = None
        else:
            fixed = np.array(to_rgb(colour_mode))
        if style_name == "line":
            _trail_line(scene, paths[:, k], fixed, cmap, width, fade_old, atom_id)
        elif style_name == "arrow":
            r_end = ctx.r_eff[row_of[atom_id]] if atom_id in row_of else 0.3
            _trail_arrow(scene, paths[:, k], fixed if fixed is not None else np.array(cmap(1.0)[:3]),
                         width, spec.get("points", 14), r_end, st, atom_id)
        else:
            sym = sym_of[atom_id]
            radius = (ctx.r_eff[row_of[atom_id]] if atom_id in row_of
                      else st.ball_scale * DATA[sym][1])
            _trail_ghosts(scene, paths[:, k], fixed if fixed is not None else np.array(st.color(sym)),
                          spec.get("count", 5), radius, st, atom_id)


def _trail_line(scene, path, fixed, cmap, width, fade_old, atom_id):
    """Every frame-to-frame step, fading with age (shows the thermal wiggle too)."""
    n_f = len(path)
    for f in range(n_f - 1):
        p0, p1 = path[f], path[f + 1]
        if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            continue
        q, d, fq = scene.project(np.array([p0, p1]))
        poly = _bond_polygon(q[0], q[1], 0.5 * width * float(scene.k(scene.t(d.mean()))) * float(fq.mean()))
        if poly is None:
            continue
        age = (n_f - 2 - f) / max(n_f - 2, 1)
        colour = fixed if fixed is not None else np.array(cmap(f / max(n_f - 2, 1))[:3])
        scene.add(d.mean(), -1, 0, _poly(poly),
                  scene.rgba(colour, float(scene.fade(scene.t(d.mean()))), 1.0 - fade_old * age),
                  (0, 0, 0, 0), 0.0, gid=f"trail-{atom_id}")
        scene.extent.extend(q)


def _simplify(path, n_points):
    """The overall route: heavy smoothing, then n_points evenly spaced along its length."""
    from .trajectory import smooth

    raw = path[np.isfinite(path).all(axis=1)]
    if len(raw) < 2:
        return raw
    pts = smooth(raw, max(3, len(raw) // 10))
    pts[0], pts[-1] = raw[0], raw[-1]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return pts[:1]
    targets = np.linspace(0, s[-1], n_points)
    return np.column_stack([np.interp(targets, s, pts[:, c]) for c in range(3)])


def _trail_arrow(scene, path, colour, width, n_points, r_end, st, atom_id):
    """A smoothed, dashed route with an arrowhead stopping at the atom's surface."""
    pts = _simplify(path, n_points)
    if len(pts) < 2:
        return
    end_dir = pts[-1] - pts[-2]
    if np.linalg.norm(end_dir) < 1e-9:
        return
    u = end_dir / np.linalg.norm(end_dir)
    tip = pts[-1] - u * r_end
    body = np.vstack([pts[:-1], tip])
    for a, b in zip(body[:-1], body[1:]):
        for p0, p1 in _dashes(a, b, 2.2 * st.dash_length, 1.6 * st.dash_gap):
            q, d, fq = scene.project(np.array([p0, p1]))
            poly = _bond_polygon(q[0], q[1], 0.5 * width * float(fq.mean()))
            if poly is not None:
                scene.add(d.mean(), -1, 0, _poly(poly), scene.rgba(colour, 0.0),
                          (0, 0, 0, 0), 0.0, gid=f"trail-{atom_id}")
                scene.extent.extend(q)
    head = 4.0 * width
    _arrow(scene, tip - u * head, tip, width, colour, 0.0, -1, 1, gid=f"trail-{atom_id}")


def _trail_ghosts(scene, path, colour, count, radius, st, atom_id):
    """Faded copies of the atom at evenly spaced earlier times (stroboscopic)."""
    good = np.flatnonzero(np.isfinite(path).all(axis=1))
    if len(good) < 2:
        return
    # space them along the smoothed route; spacing in time piles them up where the atom waits
    route = _simplify(path, count + 1)[:-1]
    pos = path[good]
    picks = np.unique([good[int(np.argmin(np.linalg.norm(pos - r, axis=1)))] for r in route])
    edge = edge_of(colour, st.atom_edgecolor, st.darken)
    for n, f in enumerate(picks):
        age = 1.0 - n / max(len(picks), 1)
        q, d, fq = scene.project(path[f][None, :])
        rad = radius * (0.65 + 0.25 * (1 - age)) * float(fq[0])
        fade = float(scene.fade(scene.t(d[0])))
        alpha = 0.18 + 0.42 * (1 - age)
        scene.add(d[0], -1, 0, _circle(tuple(q[0]), rad), scene.rgba(colour, fade, alpha),
                  scene.rgba(edge, fade, alpha), 0.4, gid=f"trail-{atom_id}")
        scene.extent.extend([q[0] - rad, q[0] + rad])
