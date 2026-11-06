"""Viewing directions and the orthographic projection onto the page.

Screen frame: x to the right, y up the page, z out of the page towards the
viewer. Larger screen z means closer to the viewer (drawn later, faded less).
"""

from __future__ import annotations

import re

import numpy as np

_AXES = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length direction")
    return v / n


def _parse_axis(spec):
    spec = spec.strip().lower()
    sign = -1.0 if spec.startswith("-") else 1.0
    key = spec.lstrip("+-")
    if key not in _AXES:
        raise ValueError(f"unknown axis {spec!r}; use x, y, z, -x, -y or -z")
    return sign * _AXES[key]


def _rot(axis, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def parse_rotation(rotate) -> np.ndarray:
    """'20x,-35y,10z' -> rotation matrix applied in the screen frame, left to
    right (rotate about the page's x axis, then its y axis, ...)."""
    if rotate is None:
        return np.eye(3)
    if not isinstance(rotate, str):
        m = np.asarray(rotate, dtype=float)
        if m.shape != (3, 3):
            raise ValueError("rotate must be a string like '20x,-35y' or a 3x3 matrix")
        return m
    m = np.eye(3)
    for tok in filter(None, (t.strip() for t in rotate.split(","))):
        match = re.fullmatch(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([xyzXYZ])", tok)
        if not match:
            raise ValueError(f"cannot parse rotation {tok!r}; use e.g. '20x'")
        m = _rot(match.group(2).lower(), float(match.group(1))) @ m
    return m


def view_matrix(view="z", *, up=None, rotate=None) -> np.ndarray:
    """Rotation whose rows are the page's right, up and out directions in
    structure coordinates.

    view: the structure direction that points out of the page at the viewer,
          as "x", "-y", ... or a 3-vector. "z" is a top view (x right, y up);
          "-y" is the usual side view (x right, z up). A 3x3 matrix is taken
          as the full rotation.
    up:   structure direction to point up the page (default z, or y when
          looking along z).
    rotate: extra rotations about the page axes, e.g. "15x,-30y".
    """
    if not isinstance(view, str) and np.asarray(view).shape == (3, 3):
        base = np.asarray(view, dtype=float)
    else:
        out = _parse_axis(view) if isinstance(view, str) else _unit(view)
        if up is None:
            up = _AXES["y"] if abs(out[2]) > 0.99 else _AXES["z"]
        up = _parse_axis(up) if isinstance(up, str) else _unit(up)
        up = up - out * (up @ out)
        if np.linalg.norm(up) < 1e-8:
            raise ValueError("'up' is parallel to the view direction")
        up = _unit(up)
        right = np.cross(up, out)
        base = np.vstack([right, up, out])
    return parse_rotation(rotate) @ base


def auto_view(positions) -> np.ndarray:
    """Principal-axes view: the longest extent runs across the page, the
    shortest points at the viewer (so planar molecules are seen face-on)."""
    p = np.asarray(positions, dtype=float)
    if len(p) < 2:
        return np.eye(3)
    p = p - p.mean(axis=0)
    _, vecs = np.linalg.eigh(p.T @ p)      # ascending variance
    right, up, out = vecs[:, 2], vecs[:, 1], vecs[:, 0]
    # make the choice deterministic: right and up lean towards +x / +z
    if right @ np.array([1, 0.01, 0.001]) < 0:
        right = -right
    if up @ np.array([0.001, 0.01, 1]) < 0:
        up = -up
    out = np.cross(right, up)
    return np.vstack([right, up, out])


def project(positions, R) -> np.ndarray:
    """(N, 3) structure coordinates -> (N, 3) page x, page y, depth."""
    return np.asarray(positions, dtype=float) @ R.T
