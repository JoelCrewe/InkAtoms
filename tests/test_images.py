"""Image regression tests (pytest-mpl).

Run with:            pytest --mpl
Regenerate baseline: pytest --mpl-generate-path=tests/baseline tests/test_images.py
Without --mpl these tests only check that the figures build.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import inkatoms as ia

ase_build = pytest.importorskip("ase.build")
compare = pytest.mark.mpl_image_compare(tolerance=6, remove_text=True, savefig_kwargs={"dpi": 80})


@compare
def test_image_styles():
    fig, axes = plt.subplots(1, 4, figsize=(6, 1.6))
    for ax, name in zip(axes, ["jmol", "cpk", "greyscale", "bw"]):
        ia.draw(ax, ase_build.molecule("C2H6SO"), view="auto", rotate="25x", style=name)
    return fig


@compare
def test_image_depth_and_cell():
    fig, ax = plt.subplots(figsize=(3, 3))
    ia.draw(ax, ase_build.mx2("MoS2", size=(3, 3, 1), vacuum=2.0), view="z", rotate="-50x",
            cell=True, shading="soft", depth_fade=0.7)
    return fig


@compare
def test_image_bond_orders_rings():
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    ia.draw(axes[0], ase_build.molecule("C4H4S"), view="auto", bond_orders=True)
    ia.draw(axes[1], ase_build.molecule("C6H6"), view="auto", rings={"aromatic": "auto"})
    return fig


@compare
def test_image_polyhedra_vectors():
    from ase.spacegroup import crystal

    from inkatoms import select

    rutile = crystal(["Ti", "O"], basis=[(0, 0, 0), (0.3, 0.3, 0)], spacegroup=136,
                     cellpar=[4.59, 4.59, 2.96, 90, 90, 90])
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    ia.draw(axes[0], select.tile(rutile, (2, 2, 2), centred=False), view="z", rotate="-25x,15y",
            polyhedra={"centers": "Ti"}, style="vesta")
    m = ase_build.molecule("H2O")
    ia.draw(axes[1], m, view="z", vectors=[[0.8, 0, 0], [0, 0.6, 0], [-0.5, -0.5, 0]])
    return fig
