"""Hover over a drawing to see atom ids; click to collect them (Jupyter %matplotlib widget,
or any interactive backend)."""

from __future__ import annotations

import numpy as np


class Picker:
    """Shows "S 2491 (x, y, z)" when hovering over an atom; clicking adds its id to
    .picked (and prints it), handy for choosing highlight= or measure() ids."""

    def __init__(self, art, *, fontsize=8, echo=True):
        self.art = art
        self.picked = []
        self.echo = echo
        ax = art.ax
        self._note = ax.annotate("", xy=(0, 0), xytext=(8, 8), textcoords="offset points", fontsize=fontsize,
                                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", lw=0.5),
                                 zorder=1000, visible=False)
        canvas = ax.figure.canvas
        self._cids = [canvas.mpl_connect("motion_notify_event", self._move),
                      canvas.mpl_connect("button_press_event", self._click)]

    def atom_at(self, x, y):
        """Row of the front-most atom under page point (x, y), or None."""
        P, r = self.art.projected, self.art.radii
        hit = np.flatnonzero(np.hypot(P[:, 0] - x, P[:, 1] - y) <= r)
        return None if len(hit) == 0 else int(hit[np.argmax(P[hit, 2])])

    def _describe(self, row):
        s = self.art.structure
        x, y, z = s.positions[row]
        return f"{s.symbols[row]} {s.ids[row]}  ({x:.2f}, {y:.2f}, {z:.2f})"

    def _move(self, event):
        if event.inaxes is not self.art.ax:
            return
        row = self.atom_at(event.xdata, event.ydata)
        visible = row is not None
        if visible:
            self._note.xy = tuple(self.art.projected[row, :2])
            self._note.set_text(self._describe(row))
        if visible or self._note.get_visible():
            self._note.set_visible(visible)
            self.art.ax.figure.canvas.draw_idle()

    def _click(self, event):
        if event.inaxes is not self.art.ax:
            return
        row = self.atom_at(event.xdata, event.ydata)
        if row is not None:
            atom_id = int(self.art.structure.ids[row])
            self.picked.append(atom_id)
            if self.echo:
                print(self._describe(row))

    def disconnect(self):
        for cid in self._cids:
            self.art.ax.figure.canvas.mpl_disconnect(cid)
        self._note.remove()


def picker(art, **kwargs) -> Picker:
    """Attach a hover/click atom picker to a drawing."""
    return Picker(art, **kwargs)
