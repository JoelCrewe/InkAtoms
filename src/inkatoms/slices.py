"""Volumetric data (charge densities, orbitals) as a slice behind a drawing."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def read_cube(path):
    """(data, atoms) from a Gaussian cube file, via ASE."""
    from ase.io.cube import read_cube_data

    return read_cube_data(str(path))


def sample_plane(art, data, cell, *, origin=(0.0, 0.0, 0.0), depth=None, resolution=0.05,
                 periodic=True):
    """Sample a 3D grid (spanning ``cell`` from ``origin``) on the page plane.

    ``depth`` places the slice along the view direction; by default it goes through the
    centre of the drawn atoms. Returns ``(X, Y, V)`` for ``contour``/``contourf``.
    """
    data = np.asarray(data, float)
    cell = np.asarray(cell, float)
    ax = art.ax
    R = art.rotation
    if depth is None:
        depth = float(art.projected[:, 2].mean())
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    nx = max(int((x1 - x0) / resolution), 2)
    ny = max(int((y1 - y0) / resolution), 2)
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    page = np.stack([X.ravel(), Y.ravel(), np.full(X.size, depth)], axis=1)
    xyz = page @ R                     # back to structure coordinates
    frac = (xyz - np.asarray(origin, float)) @ np.linalg.inv(cell)
    idx = (frac * np.array(data.shape)).T
    V = map_coordinates(data, idx, order=1, mode="grid-wrap" if periodic else "nearest")
    return X, Y, V.reshape(X.shape)


def density_slice(art, data, cell, *, origin=(0.0, 0.0, 0.0), depth=None, filled=True, levels=12,
                  cmap="Blues", alpha=0.85, resolution=0.05, periodic=True, zorder=None, **kwargs):
    """Draw a slice of volumetric data in the page plane, behind the structure.

    Use with orthographic views (perspective drawings are not matched).
    Returns the contour set; pass it to plt.colorbar for a scale.
    """
    X, Y, V = sample_plane(art, data, cell, origin=origin, depth=depth, resolution=resolution,
                           periodic=periodic)
    z = (art.collection.get_zorder() if art.collection is not None else 2) - 1 if zorder is None else zorder
    fn = art.ax.contourf if filled else art.ax.contour
    return fn(X, Y, V, levels=levels, cmap=cmap, alpha=alpha, zorder=z, **kwargs)
