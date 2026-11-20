"""Atom labels and measurement annotations (distances, angles, dihedrals)."""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc

from .styles import mix


def _luminance(rgb):
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _data_per_point(ax):
    """Å per typographic point along x for the current limits."""
    fig = ax.figure
    w_pt = ax.get_position().width * fig.get_figwidth() * 72
    return abs(np.ptp(ax.get_xlim())) / max(w_pt, 1e-9)


def _label_texts(art, labels, values, fmt):
    s = art.structure
    n = len(s)
    if isinstance(labels, str):
        if labels == "symbol":
            return dict(enumerate(s.symbols))
        if labels == "id":
            return {r: str(i) for r, i in enumerate(s.ids)}
        if labels == "index":
            return {r: str(r) for r in range(n)}
        if labels == "value":
            if values is None:
                raise ValueError("labels='value' needs color_by=")
            return {r: fmt.format(v) for r, v in enumerate(values) if np.isfinite(v)}
        raise ValueError("labels must be 'symbol', 'id', 'index', 'value', a list or {id: text}")
    if isinstance(labels, dict):
        out = {}
        for atom_id, text in labels.items():
            for r in s.index_of(atom_id):
                out[r] = str(text)
        return out
    if callable(labels):
        return {r: str(labels(s, r)) for r in range(n)}
    labels = list(labels)
    if len(labels) != n:
        raise ValueError(f"{len(labels)} labels for {n} atoms")
    return {r: str(t) for r, t in enumerate(labels) if t is not None and str(t) != ""}


def _repel(pos, sizes, anchors, iterations=80):
    """Push overlapping label boxes apart (simple pairwise relaxation)."""
    pos = pos.copy()
    for _ in range(iterations):
        moved = False
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                d = pos[b] - pos[a]
                ov = (sizes[a] + sizes[b]) / 2 - np.abs(d)
                if ov[0] > 0 and ov[1] > 0:
                    axis = int(ov[1] < ov[0])
                    push = 0.5 * ov[axis] + 1e-3
                    sign = 1.0 if d[axis] >= 0 else -1.0
                    pos[a, axis] -= sign * push
                    pos[b, axis] += sign * push
                    moved = True
        pos += 0.02 * (anchors - pos)
        if not moved:
            break
    return pos


def _draw_labels(art, labels, values, base, fade_atom, kw, zorder):
    ax = art.ax
    fmt = kw.pop("fmt", "{:.2f}")
    offset = kw.pop("offset", None)
    repel = kw.pop("repel", offset is not None)
    include_images = kw.pop("include_images", False)
    colour = kw.pop("color", kw.pop("colour", "auto"))
    kw.setdefault("fontsize", 6.5)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    texts = _label_texts(art, labels, values, fmt)
    is_image = art.structure.arrays.get("_image")
    if is_image is not None and not include_images:
        texts = {r: t for r, t in texts.items() if not is_image[r]}
    if not texts:
        return []
    rows = np.array(sorted(texts))
    anchors = art.projected[rows, :2]
    pos = anchors + (np.asarray(offset, float) if offset is not None else 0.0)
    if repel:
        dpp = _data_per_point(ax)
        fs = kw["fontsize"]
        sizes = np.array([[0.62 * fs * max(len(texts[r]), 1) * dpp + 0.1, 1.25 * fs * dpp] for r in rows])
        pos = _repel(pos, sizes, pos)
    bg = np.asarray(art._scene.bg)
    out = []
    for r, p, anchor in zip(rows, pos, anchors):
        c = colour
        if isinstance(colour, str) and colour == "auto":
            if offset is None:
                face = mix(base[r], bg, fade_atom[r])
                c = "#111111" if _luminance(face) > 0.5 else "#ffffff"
            else:
                c = "#111111" if _luminance(bg) > 0.5 else "#eeeeee"
        if offset is not None and np.hypot(*(p - anchor)) > art.radii[r]:
            out.append(ax.annotate(texts[r], xy=tuple(anchor), xytext=tuple(p), color=c, zorder=zorder + 3,
                                   arrowprops=dict(arrowstyle="-", lw=0.4, color=c, shrinkA=0,
                                                   shrinkB=0.5), **kw))
        else:
            out.append(ax.text(p[0], p[1], texts[r], color=c, zorder=zorder + 3, **kw))
    return out


def measure(art, ids, *, fmt=None, color="#333333", fontsize=7, offset=0.35, lw=0.7,
            dashed=True, zorder=None):
    """Annotate a distance (2 ids), angle (3 ids, at the middle atom) or dihedral
    (4 ids) on a drawing, using the true 3D value. Returns the artists added."""
    ax = art.ax
    ids = list(ids)
    if len(ids) not in (2, 3, 4):
        raise ValueError("measure needs 2 (distance), 3 (angle) or 4 (dihedral) atom ids")
    z = (art.collection.get_zorder() if art.collection is not None else 2) + 4 if zorder is None else zorder
    p3 = np.array([art.position(i) for i in ids])
    p2 = np.array([art.xy(i) for i in ids])
    r2 = np.array([art.radius(i) for i in ids])
    ls = (0, (2.5, 1.8)) if dashed else "-"
    out = []
    if len(ids) == 2:
        value = np.linalg.norm(p3[1] - p3[0])
        text = (fmt or "{:.2f} Å").format(value)
        d = p2[1] - p2[0]
        L = np.hypot(*d)
        if L < 1e-9:
            raise ValueError("the two atoms overlap on the page in this view")
        u = d / L
        a, b = p2[0] + u * r2[0], p2[1] - u * r2[1]
        out.append(ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=z)))
        perp = np.array([-u[1], u[0]])
        if perp[1] < 0:
            perp = -perp
        m = 0.5 * (p2[0] + p2[1]) + perp * offset
        ang = np.degrees(np.arctan2(u[1], u[0]))
        if ang > 90:
            ang -= 180
        if ang < -90:
            ang += 180
        out.append(ax.text(*m, text, rotation=ang, rotation_mode="anchor", ha="center", va="bottom",
                           fontsize=fontsize, color=color, zorder=z))
    elif len(ids) == 3:
        v1, v2 = p3[0] - p3[1], p3[2] - p3[1]
        value = np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
        text = (fmt or "{:.1f}°").format(value)
        c = p2[1]
        a1 = np.degrees(np.arctan2(*(p2[0] - c)[::-1]))
        a2 = np.degrees(np.arctan2(*(p2[2] - c)[::-1]))
        span = (a2 - a1) % 360
        if span > 180:
            a1, a2 = a2, a1
            span = 360 - span
        rad = max(1.6 * r2[1], 0.45 * min(np.hypot(*(p2[0] - c)), np.hypot(*(p2[2] - c))))
        out.append(ax.add_patch(Arc(tuple(c), 2 * rad, 2 * rad, theta1=a1, theta2=a1 + span,
                                    color=color, lw=lw, ls=ls, zorder=z)))
        mid = np.radians(a1 + span / 2)
        tp = c + (rad + offset) * np.array([np.cos(mid), np.sin(mid)])
        out.append(ax.text(*tp, text, ha="center", va="center", fontsize=fontsize, color=color, zorder=z))
    else:
        b1, b2, b3 = p3[1] - p3[0], p3[2] - p3[1], p3[3] - p3[2]
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        value = np.degrees(np.arctan2(m1 @ n2, n1 @ n2))
        text = (fmt or "φ = {:.1f}°").format(value)
        for a, b in ((0, 1), (2, 3)):
            out.append(ax.add_line(Line2D(p2[[a, b], 0], p2[[a, b], 1], color=color, lw=lw * 0.8,
                                          ls=ls, zorder=z)))
        m = 0.5 * (p2[1] + p2[2]) + np.array([0, offset + r2[1:3].max()])
        out.append(ax.text(*m, text, ha="center", va="bottom", fontsize=fontsize, color=color, zorder=z))
    return out
