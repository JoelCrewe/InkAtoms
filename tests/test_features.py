"""Tests for the 0.2 features: colouring, vectors, polyhedra, rings, contacts,
markers, images, perspective, slices, inputs, figures, CLI, picker."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import inkatoms as ia
from inkatoms import select
from inkatoms.structure import Structure

ase_build = pytest.importorskip("ase.build")
from ase import Atoms  # noqa: E402
from ase.constraints import FixAtoms  # noqa: E402
from ase.spacegroup import crystal  # noqa: E402

WATER_DIMER = Atoms("OH2OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0],
                                         [2.9, 0, 0], [3.4, 0.8, 0], [3.4, -0.8, 0]])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def n_pieces(art):
    return len(art.collection.get_paths()) if art.collection is not None else len(art.patches)


def rutile():
    return crystal(["Ti", "O"], basis=[(0, 0, 0), (0.3, 0.3, 0)], spacegroup=136,
                   cellpar=[4.59, 4.59, 2.96, 90, 90, 90])


# ---------------------------------------------------------------- structure data

def test_ase_arrays_constraints_and_charges_kept():
    m = ase_build.molecule("H2O")
    m.set_initial_charges([-0.8, 0.4, 0.4])
    m.set_constraint(FixAtoms(indices=[0]))
    s = ia.as_structure(m)
    assert np.allclose(s.arrays["initial_charges"], [-0.8, 0.4, 0.4])
    assert list(s.arrays["fixed"]) == [True, False, False]
    sub = s[[0, 2]]
    assert list(sub.arrays["fixed"]) == [True, False]


def test_pymatgen_partial_occupancy():
    core = pytest.importorskip("pymatgen.core")
    ps = core.Structure(core.Lattice.cubic(4.0), [{"Fe": 0.5, "Co": 0.5}, {"O": 1.0}],
                        [[0, 0, 0], [0.5, 0.5, 0.5]])
    s = ia.as_structure(ps)
    assert s.occupancy[0] and {sym for sym, _ in s.occupancy[0]} == {"Fe", "Co"}
    assert s.occupancy[1] is None
    fig, ax = plt.subplots()
    plain = n_pieces(ia.draw(ax, (s.symbols, s.positions), bonds=False))
    pies = n_pieces(ia.draw(ax, s, bonds=False, pbc_bonds=False))
    assert pies == plain + 3       # two wedges and an outline on the mixed site


def test_rdkit_and_smiles_bond_orders():
    pytest.importorskip("rdkit")
    s = ia.from_smiles("C=CC#N")
    assert sorted(s.bond_orders.values()).count(3.0) == 1
    assert sorted(s.bond_orders.values()).count(2.0) == 1
    fig, ax = plt.subplots()
    n1 = n_pieces(ia.draw(ax, s, view="auto"))
    n2 = n_pieces(ia.draw(ax, s, view="auto", bond_orders=True))
    assert n2 == n1 + 2 * 1 + 2 * 2     # C=C adds one line pair of halves, C#N two


def test_mdanalysis_input():
    mda = pytest.importorskip("MDAnalysis")
    u = mda.Universe.empty(3, trajectory=True)
    u.add_TopologyAttr("names", ["OW", "HW1", "HW2"])
    u.atoms.positions = ase_build.molecule("H2O").positions
    assert list(ia.as_structure(u).symbols) == ["O", "H", "H"]


def test_cell_from_parameters_roundtrip():
    cell = ia.cell_from_parameters(3, 4, 5, 80, 95, 110)
    lengths = np.linalg.norm(cell, axis=1)
    assert np.allclose(lengths, [3, 4, 5])
    cos_gamma = cell[0] @ cell[1] / (3 * 4)
    assert np.isclose(np.degrees(np.arccos(cos_gamma)), 110)


# ---------------------------------------------------------------- selection

def test_images_boundary_and_make_whole():
    cu = ase_build.bulk("Cu", "fcc", a=3.6, cubic=True)
    b = select.boundary(cu)
    assert len(b) == 14 and b.arrays["_image"].sum() == 10
    assert len(set(b.ids)) == len(b)
    mol = ase_build.molecule("C6H6")
    mol.set_cell([6, 6, 6])
    mol.pbc = True
    mol.positions += [5.5, 3, 3]
    mol.wrap()
    whole = select.make_whole(mol)
    assert np.ptp(whole.positions[:, 0]) < 5.5


# ---------------------------------------------------------------- colour, vectors

def test_color_by_and_colorbar():
    fig, ax = plt.subplots()
    art = ia.draw(ax, ase_build.molecule("CH3CH2OH"), view="auto", color_by="z", cmap="plasma")
    assert art.mappable is not None and len(art.values) == 9
    cb = ia.colorbar(art, label="z")
    assert cb.ax.get_ylabel() == "z"
    with pytest.raises(KeyError):
        ia.draw(ax, WATER_DIMER, color_by="nope")


def test_vectors_add_arrows():
    fig, ax = plt.subplots()
    m = ase_build.molecule("H2O")
    base = n_pieces(ia.draw(ax, m, view="z"))
    v = np.zeros((3, 3))
    v[0] = [1, 0, 0]
    art = ia.draw(ax, m, view="z", vectors=v)
    assert n_pieces(art) == base + 2     # shaft + head


# ---------------------------------------------------------------- polyhedra, rings, contacts

def test_polyhedra_rutile_octahedra():
    fig, ax = plt.subplots()
    s = select.tile(rutile(), (2, 2, 2), centred=False)
    assert (ia.coordination(s)[s.symbols == "Ti"] == 6).all()      # no Ti-Ti bonds
    plain = n_pieces(ia.draw(ax, s, view="z"))
    poly = ia.draw(ax, s, view="z", polyhedra={"centers": "Ti"})
    n_ti = int((s.symbols == "Ti").sum())
    assert n_pieces(poly) == plain - n_ti * 6 * 2 + n_ti * 8    # bonds hidden, 8 faces each
    assert ia.scene._hull_faces(np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                                           [0, 0, 1], [0, 0, -1]], float)).__len__() == 8


def test_hull_faces_merge_coplanar_and_flat():
    cube = np.array(list(np.ndindex(2, 2, 2)), float)
    faces = ia.scene._hull_faces(cube)
    assert len(faces) == 6 and all(len(f) == 4 for f in faces)
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    assert len(ia.scene._hull_faces(square)) == 1


def test_rings_and_aromatic_circle():
    fig, ax = plt.subplots()
    m = ase_build.molecule("C6H6")
    plain = n_pieces(ia.draw(ax, m, view="auto"))
    ring = n_pieces(ia.draw(ax, m, view="auto", rings={"aromatic": "auto"}))
    assert ring == plain + 2


def test_hbonds_and_bond_styles():
    assert ia.find_hbonds(WATER_DIMER) == [(1, 3)]
    fig, ax = plt.subplots()
    plain = n_pieces(ia.draw(ax, WATER_DIMER, view="z"))
    hb = n_pieces(ia.draw(ax, WATER_DIMER, view="z", contacts="hbonds"))
    assert hb > plain
    hidden = n_pieces(ia.draw(ax, WATER_DIMER, view="z", bond_styles={(3, 4): "hidden"}))
    assert hidden == plain - 2
    ethane = ase_build.molecule("C2H6")
    p0 = n_pieces(ia.draw(ax, ethane, view="auto"))
    dashed = n_pieces(ia.draw(ax, ethane, view="auto", bond_styles={(0, 1): "breaking"}))
    assert dashed > p0


# ---------------------------------------------------------------- markers, cell extras, images

def test_constraints_vacancies_ghosts_planes():
    slab = ase_build.fcc111("Pt", size=(2, 2, 2), vacuum=5.0)
    slab.set_constraint(FixAtoms(indices=[0, 1]))
    fig, ax = plt.subplots()
    plain = n_pieces(ia.draw(ax, slab, view="-y", pbc_bonds=False))
    fixed = n_pieces(ia.draw(ax, slab, view="-y", pbc_bonds=False, constraints=True))
    assert fixed == plain + 4
    vac = n_pieces(ia.draw(ax, slab, view="-y", pbc_bonds=False, vacancies=[[0, 0, 12]]))
    assert vac == plain + 1
    cu = ase_build.bulk("Cu", "fcc", a=3.6, cubic=True)
    art = ia.draw(ax, cu, ghosts=1.0, cell=True, cell_vectors=True, planes=[(1, 1, 1)])
    assert len(art.structure) == 14 and [t.get_text() for t in art.extras] == ["a", "b", "c"]
    with pytest.raises(ValueError):
        ia.draw(ax, WATER_DIMER, cell=True)


def test_plane_polygon_shapes():
    cell, o = np.eye(3), np.zeros(3)
    assert len(ia.scene._plane_polygon(cell, o, (1, 1, 1))) == 3
    assert len(ia.scene._plane_polygon(cell, o, (0, 0, 1), 0.5)) == 4
    assert len(ia.scene._plane_polygon(cell, o, (1, 1, 1), 1.5)) == 6


def test_perspective_and_occlusion():
    fig, ax = plt.subplots()
    s = (["C", "C"], [[0, 0, 0], [0, 0, 4]])
    art = ia.draw(ax, s, view="z", perspective=10.0, bonds=False, depth_shrink=0.0)
    ortho = ia.draw(ax, s, view="z", bonds=False, depth_shrink=0.0)
    assert np.isclose(ortho.radii[0], ortho.radii[1])
    assert not np.isclose(art.radii[0], art.radii[1])
    cl = select.around(ase_build.bulk("Au", "fcc", a=4.08).repeat((4, 4, 4)), at=None, radius=5)
    a1 = ia.draw(ax, cl, mode="spacefill", occlusion=1.0)
    assert a1.collection is not None


def test_labels_measure_and_editable(tmp_path):
    fig, ax = plt.subplots()
    m = ase_build.molecule("CH3COOH")
    art = ia.draw(ax, m, view="auto", labels="symbol", label_kw={"offset": (0.4, 0.4)})
    assert len(art.labels) == len(m)
    d = ia.measure(art, [0, 1])
    assert d[-1].get_text().endswith("Å")
    ang = ia.measure(art, [1, 0, 2])
    assert 115 < float(ang[-1].get_text().rstrip("°")) < 130
    dih = ia.measure(art, [1, 0, 4, 5])
    assert dih[-1].get_text().startswith("φ")
    fig2, ax2 = plt.subplots()
    ia.draw(ax2, ase_build.molecule("H2O"), view="z", editable=True)
    out = tmp_path / "w.svg"
    fig2.savefig(out)
    text = out.read_text()
    assert text.count('id="atom-') == 3 and text.count('id="bond-') >= 2


def test_rasterize_auto_threshold(monkeypatch):
    fig, ax = plt.subplots()
    monkeypatch.setattr(ia.render, "AUTO_RASTERIZE", 5)
    assert ia.draw(ax, ase_build.molecule("C6H6"), view="auto").collection.get_rasterized()
    monkeypatch.setattr(ia.render, "AUTO_RASTERIZE", 10**6)
    assert not ia.draw(ax, ase_build.molecule("C6H6"), view="auto").collection.get_rasterized()


def test_new_presets():
    for name in ("vesta", "colourblind", "colorblind", "dark"):
        st = ia.style(name)
        assert len(st.color("S")) == 3
    assert ia.style("vesta").color("S") != ia.style("jmol").color("S")
    assert ia.figsize("rsc", "double", 0.5) == (6.73, 3.365)


# ---------------------------------------------------------------- slices

def test_density_slice_samples_the_grid():
    fig, ax = plt.subplots()
    cell = np.diag([4.0, 4.0, 4.0])
    data = np.zeros((20, 20, 20))
    data[10, :, :] = 1.0          # a sheet at x = 2 Å
    art = ia.draw(ax, (["H"], [[2.0, 2.0, 2.0]]), view="z")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    X, Y, V = ia.sample_plane(art, data, cell, resolution=0.1)
    col = V[:, np.argmin(np.abs(X[0] - 2.0))]
    assert col.mean() > 0.5 and V[:, 0].mean() < 0.1
    ia.density_slice(art, data, cell)


# ---------------------------------------------------------------- figures

def test_inset_panels_animate():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    iax, art = ia.inset(ax, ase_build.molecule("H2O"), (0.5, 0.5), size=0.3)
    assert not iax.get_in_layout()
    fig, axes = plt.subplots(1, 3)
    frames = [ase_build.molecule("H2O") for _ in range(3)]
    frames[2].positions[1] += [0.5, 0, 0]
    arts = ia.panels(axes, frames, view="z")
    assert len({ax_.get_xlim() for ax_ in axes}) == 1
    anim = ia.animate(ase_build.molecule("C6H6"), spin=90, view="auto")
    anim._func(2)       # draw a frame without saving
    anim._draw_was_started = True   # silence "deleted without rendering"
    assert len(arts) == 3


def test_axes_triad_and_picker():
    fig, ax = plt.subplots()
    art = ia.draw(ax, ase_build.molecule("H2O"), view="z")
    assert len(ia.axes_triad(ax, art)) == 6
    p = ia.picker(art, echo=False)
    x, y = art.xy(0)
    assert p.atom_at(x, y) == 0
    assert p.atom_at(x + 50, y) is None
    p.disconnect()


def test_cli(tmp_path):
    from ase.io import write

    from inkatoms import cli

    f = tmp_path / "m.xyz"
    write(f, ase_build.molecule("CH3CONH2"))
    out = tmp_path / "m.svg"
    cli.main([str(f), "-o", str(out), "--bond-orders", "--style", "cpk", "--legend", "--labels", "symbol"])
    assert out.exists() and out.stat().st_size > 1000


def test_explicit_bond_list():
    fig, ax = plt.subplots()
    s = (["C", "C", "C"], [[0, 0, 0], [1.5, 0, 0], [6, 0, 0]])
    art = ia.draw(ax, s, view="z", bonds=[(0, 2)])          # far apart, but bonded because we say so
    assert n_pieces(art) == 3 + 2
    with pytest.raises(KeyError):
        ia.draw(ax, s, bonds=[(0, 9)])


def test_fade_distance_is_absolute():
    fig, ax = plt.subplots()
    s = (["C", "C"], [[0, 0, 0], [3, 0, 1.0]])          # 1 Å apart in depth
    rel = ia.draw(ax, s, view="z", bonds=False)
    absf = ia.draw(ax, s, view="z", bonds=False, fade_distance=10.0)
    # normalised: the back atom is fully faded; with a 10 Å fade length only a tenth
    assert rel.radii[0] < absf.radii[0] < absf.radii[1]


def test_per_element_edge_colours():
    fig, ax = plt.subplots()
    art = ia.draw(ax, ase_build.molecule("H2O"), view="z", bonds=False, depth_fade=0.0,
                  edge_colors={"H": "#ff0000"})
    edges = art.collection.get_edgecolor()
    assert sum(np.allclose(e[:3], (1, 0, 0)) for e in edges) == 2


def test_depth_cue_toggle():
    fig, ax = plt.subplots()
    s = (["C", "C"], [[0, 0, 0], [3, 0, 2.0]])
    on = ia.draw(ax, s, view="z", bonds=False)
    off = ia.draw(ax, s, view="z", bonds=False, depth_cue=False)
    assert on.radii[0] < on.radii[1]
    assert np.isclose(off.radii[0], off.radii[1])
    faces = off.collection.get_facecolor()
    assert np.allclose(faces[0], faces[1])


def test_use_collection_faces_is_safe(tmp_path):
    out = ia.use_collection_faces(cache_dir=tmp_path)
    assert isinstance(out, list)          # empty off macOS or on matplotlib >= 3.11


def test_layers_override_depth_order():
    fig, ax = plt.subplots()
    s = (["Mo", "S"], [[0, 0, 0], [0.3, 0, 3.0]])      # S in front of Mo
    by_depth = ia.draw(ax, s, view="z", bonds=False, depth_cue=False)
    layered = ia.draw(ax, s, view="z", bonds=False, depth_cue=False, layers={"S": 0, "Mo": 1})
    mo = ia.style("jmol").color("Mo")
    assert not np.allclose(by_depth.collection.get_facecolor()[-1][:3], mo)
    assert np.allclose(layered.collection.get_facecolor()[-1][:3], mo)
