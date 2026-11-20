"""Insets, panels and animations."""

from __future__ import annotations

import numpy as np

from .render import draw
from .structure import as_structure
from .view import auto_view, view_matrix


def inset(ax, atoms, xy, size=0.25, *, coords="data", frame=False, **draw_kwargs):
    """Draw a structure in a small inset centred on ``xy``.

    ``xy`` is in data coordinates (or axes fraction with ``coords="axes"``), so set the
    axes limits first. ``size`` is the inset size as a fraction of ``ax``. The inset is
    excluded from tight/constrained layout. Returns ``(inset_ax, artist)``.
    """
    w, h = (size, size) if np.isscalar(size) else size
    if coords == "data":
        px, py = ax.transAxes.inverted().transform(ax.transData.transform(xy))
    elif coords == "axes":
        px, py = xy
    else:
        raise ValueError("coords must be 'data' or 'axes'")
    iax = ax.inset_axes([px - w / 2, py - h / 2, w, h])
    iax.patch.set_alpha(0.0)
    iax.set_in_layout(False)
    art = draw(iax, atoms, **draw_kwargs)
    if frame:
        iax.set_axis_on()
        iax.set_xticks([])
        iax.set_yticks([])
    return iax, art


def _frame_view(structs, view, up, rotate):
    if isinstance(view, str) and view == "auto":
        view = auto_view(structs[0].positions)
    return view_matrix(view, up=up, rotate=rotate), view


def panels(axes, frames, *, view="z", up=None, rotate=None, same_limits=True, same_depth=True,
           pad=0.5, **draw_kwargs):
    """Draw frames into a row of axes with the same view, depth fade and limits."""
    axes = list(np.ravel(axes))
    structs = [as_structure(f) for f in frames]
    if len(axes) < len(structs):
        raise ValueError(f"{len(structs)} frames but only {len(axes)} axes")
    R, view = _frame_view(structs, view, up, rotate)
    if same_depth and "depth_range" not in draw_kwargs:
        d = np.concatenate([(st.positions @ R.T)[:, 2] for st in structs])
        draw_kwargs["depth_range"] = (float(d.min()), float(d.max()))
    arts = [draw(ax, st, view=view, up=up, rotate=rotate, pad=pad, **draw_kwargs)
            for ax, st in zip(axes, structs)]
    if same_limits:
        b = np.array([a.bounds for a in arts])
        xl = (b[:, 0].min() - pad, b[:, 1].max() + pad)
        yl = (b[:, 2].min() - pad, b[:, 3].max() + pad)
        for ax in axes[:len(arts)]:
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
    return arts


def animate(frames, *, fig=None, ax=None, spin=None, n_frames=None, interval=80, view="z",
            up=None, rotate=None, pad=0.5, title=None, **draw_kwargs):
    """FuncAnimation of a trajectory, or of one structure spinning (``spin`` degrees per frame).

    ``title`` is an optional function of the frame index. Save with
    ``anim.save("movie.gif", writer="pillow")``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if spin is not None:
        single = frames if not isinstance(frames, (list, tuple)) else frames[0]
        n_frames = n_frames or int(round(360 / abs(spin)))
        structs = [as_structure(single)] * n_frames
        base_rot = rotate or ""
        rotations = [",".join(filter(None, [base_rot, f"{spin * k}y"])) for k in range(n_frames)]
    else:
        structs = [as_structure(f) for f in frames]
        rotations = [rotate] * len(structs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4)) if fig is None else (fig, fig.add_subplot())
    fig = ax.figure
    R, view = _frame_view(structs, view, up, rotate)
    if spin is not None:
        # fixed limits and depth range from the bounding sphere
        c = structs[0].positions.mean(axis=0)
        rad = np.linalg.norm(structs[0].positions - c, axis=1).max() + 1.5
        draw_kwargs.setdefault("depth_range", (float((c @ R.T)[2] - rad), float((c @ R.T)[2] + rad)))
        cx, cy = (c @ R.T)[:2]
        limits = ((cx - rad - pad, cx + rad + pad), (cy - rad - pad, cy + rad + pad))
    else:
        d = np.concatenate([(st.positions @ R.T)[:, 2] for st in structs])
        draw_kwargs.setdefault("depth_range", (float(d.min()), float(d.max())))
        xy = np.concatenate([(st.positions @ R.T)[:, :2] for st in structs])
        limits = ((xy[:, 0].min() - 1.5 - pad, xy[:, 0].max() + 1.5 + pad),
                  (xy[:, 1].min() - 1.5 - pad, xy[:, 1].max() + 1.5 + pad))
    state = {"art": None}

    def update(k):
        if state["art"] is not None:
            state["art"].remove()
        state["art"] = draw(ax, structs[k], view=view, up=up, rotate=rotations[k],
                            autoscale=False, **draw_kwargs)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        if title is not None:
            ax.set_title(title(k))
        return [state["art"].collection] if state["art"].collection is not None else state["art"].patches

    update(0)
    return FuncAnimation(fig, update, frames=len(structs), interval=interval, blit=False)
