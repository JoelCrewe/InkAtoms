import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import inkatoms as ia
from inkatoms import select
from inkatoms.structure import Structure

WATER = (["O", "H", "H"], [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------- structure

def test_as_structure_from_arrays():
    s = ia.as_structure(WATER)
    assert len(s) == 3 and list(s.ids) == [0, 1, 2]
    assert s.cell is None and not s.pbc.any()


def test_as_structure_from_ase():
    ase = pytest.importorskip("ase.build")
    atoms = ase.bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.new_array("uid", np.arange(100, 104))
    s = ia.as_structure(atoms, ids="uid")
    assert s.pbc.all() and s.cell.shape == (3, 3)
    assert list(s.ids) == [100, 101, 102, 103]


def test_unknown_symbol_rejected():
    with pytest.raises(ValueError):
        Structure(["Xx"], [[0, 0, 0]])


# ---------------------------------------------------------------- view

@pytest.mark.parametrize("view", ["x", "-x", "y", "-y", "z", "-z", (1, 1, 1)])
def test_view_matrix_is_proper_rotation(view):
    R = ia.view_matrix(view, rotate="20x,-35y,5z")
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_side_view_axes():
    R = ia.view_matrix("-y")
    assert np.allclose(R[0], [1, 0, 0])   # x to the right
    assert np.allclose(R[1], [0, 0, 1])   # z up
    R = ia.view_matrix("z")
    assert np.allclose(R[0], [1, 0, 0]) and np.allclose(R[1], [0, 1, 0])


# ---------------------------------------------------------------- bonds

def test_water_bonds():
    i, j, shift = ia.find_bonds(WATER)
    pairs = {tuple(sorted(p)) for p in zip(i, j)}
    assert pairs == {(0, 1), (0, 2)}
    assert not shift.any()


def test_cutoff_override_and_disable():
    i, _, _ = ia.find_bonds(WATER, cutoffs={"O-H": 0})
    assert len(i) == 0


def test_periodic_bonds_simple_cubic():
    s = Structure(["C"], [[0, 0, 0]], cell=np.eye(3) * 1.5, pbc=True)
    i, j, shift = ia.find_bonds(s)
    # each atom bonds to its 6 images; each bond is counted once
    assert len(i) == 3
    assert {tuple(np.abs(x)) for x in shift} == {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
    assert len(ia.find_bonds(s, pbc=False)[0]) == 0


def test_coordination_fcc():
    ase = pytest.importorskip("ase.build")
    atoms = ase.bulk("Cu", "fcc", a=3.6, cubic=True)
    assert (ia.coordination(atoms) == 12).all()


# ---------------------------------------------------------------- select

def test_select_keeps_ids_and_drops_pbc():
    s = Structure(["C"] * 5, [[x, 0, 0] for x in range(5)], cell=np.eye(3) * 10, pbc=True,
                  ids=[10, 11, 12, 13, 14])
    cut = select.box(s, [-0.5, -1, -1], [2.5, 1, 1], keep=[14])
    assert list(cut.ids) == [10, 11, 12, 14]
    assert not cut.pbc.any()
    sl = select.slab(s, axis="x", half_width=1.0, at=[12])
    assert list(sl.ids) == [11, 12, 13]
    ar = select.around(s, [10], radius=1.5)
    assert list(ar.ids) == [10, 11]


def test_tile_centres_original_cell():
    s = Structure(["C"], [[0.5, 0.5, 0.5]], cell=np.eye(3), pbc=True)
    t = select.tile(s, (3, 3, 1))
    assert len(t) == 9 and np.allclose(t.cell, np.diag([3, 3, 1]))
    assert np.allclose(t.positions.mean(axis=0), [0.5, 0.5, 0.5])
    assert np.allclose(t.origin, [-1, -1, 0])


def test_drop_unbonded():
    s = ia.as_structure((["O", "H", "H", "C"], WATER[1] + [[9, 9, 9]]))
    assert list(select.drop_unbonded(s).ids) == [0, 1, 2]


# ---------------------------------------------------------------- styles

@pytest.mark.parametrize("name", ["jmol", "cpk", "greyscale", "bw", "grayscale", "mono"])
def test_presets_resolve(name):
    st = ia.style(name)
    for sym in ("H", "C", "O", "S", "Mo", "Og"):
        assert len(st.color(sym)) == 3


def test_standard_colours():
    assert ia.style("jmol").color("O")[0] > 0.9      # red
    assert ia.style("jmol").color("S")[2] < 0.3      # yellow
    assert ia.style("cpk").color("C") < (0.2, 0.2, 0.2)
    assert ia.style("bw").color("Mo") == (1.0, 1.0, 1.0)
    grey = ia.style("greyscale").color("O")
    assert grey[0] == grey[1] == grey[2]


def test_style_overrides():
    st = ia.style("cpk", shading="soft", colors={"Mo": "#123456"})
    assert st.shading == "soft" and st.color("C") == ia.style("cpk").color("C")
    with pytest.raises(TypeError):
        ia.style("jmol", not_a_field=1)


# ---------------------------------------------------------------- render

@pytest.mark.parametrize("mode", ["ballstick", "spacefill", "sticks"])
@pytest.mark.parametrize("shading", ["flat", "glossy", "soft"])
def test_draw_smoke(mode, shading):
    fig, ax = plt.subplots()
    art = ia.draw(ax, WATER, view="-z", mode=mode, shading=shading, highlight={0: "red"})
    fig.canvas.draw()
    assert len(art.collection.get_paths()) >= 3
    x, y = art.xy(0)
    assert np.isclose(x, 0) and np.isclose(y, 0)


def test_draw_order_is_back_to_front():
    # two atoms on the view axis: the near one must be drawn last
    fig, ax = plt.subplots()
    art = ia.draw(ax, (["C", "O"], [[0, 0, 0], [0, 0, 5]]), view="z", bonds=False)
    faces = art.collection.get_facecolor()
    assert np.allclose(faces[-1][:3], ia.style("jmol").color("O"))


def test_highlight_is_not_faded():
    fig, ax = plt.subplots()
    s = (["C", "C"], [[0, 0, 0], [3, 0, 3]])
    plain = ia.draw(ax, s, view="z", bonds=False)
    hl = ia.draw(ax, s, view="z", bonds=False, highlight=[0])
    assert plain.radius(0) < hl.radius(0)


def test_periodic_cell_draws():
    ase = pytest.importorskip("ase.build")
    atoms = ase.mx2("MoS2", size=(2, 2, 1), vacuum=3)
    fig, ax = plt.subplots()
    art = ia.draw(ax, atoms, view="z", cell=True, style="bw")
    fig.canvas.draw()
    assert art.symbols == ["Mo", "S"]
    ia.legend(ax, art)
    ia.scale_bar(ax, 2)


def test_tile_ids_unique_and_home_copy_keeps_ids():
    s = Structure(["C", "O"], [[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]], cell=np.eye(3), pbc=True)
    t = select.tile(s, (3, 3, 1))
    assert len(set(t.ids)) == len(t)
    home = t.index_of([0, 1])
    assert np.allclose(t.positions[home], s.positions)


def test_no_metal_metal_bonds_in_mos2():
    ase = pytest.importorskip("ase.build")
    s = ia.as_structure(ase.mx2("MoS2", size=(2, 2, 1), vacuum=3))
    i, j, _ = ia.find_bonds(s)
    assert not np.any((s.symbols[i] == "Mo") & (s.symbols[j] == "Mo"))
    assert (ia.coordination(s)[s.symbols == "Mo"] == 6).all()


def test_cutoff_keys_all_forms():
    sp = ["Mo", "S"]
    for key in ("Mo-Mo", ("Mo", "Mo"), frozenset(("Mo", "Mo"))):
        assert ia.cutoff_matrix(sp, {key: 3.1})[0, 0] == 3.1
    assert ia.cutoff_matrix(sp, {frozenset(("S", "Mo")): 2.85})[0, 1] == 2.85


def test_highlight_front_draws_highlight_last():
    fig, ax = plt.subplots()
    s = (["C", "O"], [[0, 0, 0], [0, 0, 5]])        # C behind O
    art = ia.draw(ax, s, view="z", bonds=False, highlight={0: "red"}, highlight_front=True)
    assert np.allclose(art.collection.get_facecolor()[-1][:3], (1, 0, 0))


# ---------------------------------------------------------------- bond orders and scale

@pytest.mark.parametrize("name, expected", [
    ("C2H4", {2: 1}), ("C2H2", {3: 1}), ("C6H6", {}), ("HCN", {3: 1}), ("CO2", {2: 2}),
    ("C2H6SO", {2: 1}), ("C4H4S", {2: 2}), ("C5H5N", {}), ("CH3COOH", {2: 1}), ("N2", {3: 1}),
])
def test_bond_orders_on_reference_molecules(name, expected):
    ase = pytest.importorskip("ase.build")
    orders = ia.bond_orders(ase.molecule(name))
    counts = {o: int((orders == o).sum()) for o in (2, 3) if (orders == o).any()}
    assert counts == expected


def test_bond_orders_drawn_and_manual():
    ase = pytest.importorskip("ase.build")
    mol = ase.molecule("C2H4")
    fig, ax = plt.subplots()
    n1 = len(ia.draw(ax, mol, view="auto").collection.get_paths())
    n2 = len(ia.draw(ax, mol, view="auto", bond_orders=True).collection.get_paths())
    assert n2 == n1 + 2          # the C=C gets a second pair of half-bonds
    n3 = len(ia.draw(ax, mol, view="auto", bond_orders={(0, 1): 3}).collection.get_paths())
    assert n3 == n1 + 4


def test_set_scale_and_structure_figure():
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes([0, 0, 1, 1])
    ia.draw(ax, WATER, view="z", scale=0.5)
    assert np.isclose(np.ptp(ax.get_xlim()), 8.0) and np.isclose(np.ptp(ax.get_ylim()), 6.0)
    fig, ax, art = ia.structure_figure(WATER, inches_per_angstrom=1.0, pad=0.5, view="z")
    x0, x1, y0, y1 = art.bounds
    assert np.isclose(fig.get_figwidth(), x1 - x0 + 1.0)
    assert np.isclose(np.ptp(ax.get_xlim()), fig.get_figwidth())
