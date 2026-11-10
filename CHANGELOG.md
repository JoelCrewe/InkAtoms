# Changelog

## 0.3.0 (unreleased)

- Beginner tutorial notebook with interactive examples (`examples/tutorial.ipynb`,
  built by `examples/build_tutorial.py`); `inkatoms[tutorial]` extra
- Trajectories: `as_frames`, `track`, `unwrap`, `mobile`
- Events: `find_events` / `find_reactions` (debounced bond changes, insertions and
  removals), `group_reactions`, `event_strip`
- Motion trails: `draw(..., trails=dict(frames=, ids=, color="time"))`
- Comparisons: `kabsch`, `match` (ids / distance / environment / hybrid), `compare`,
  `draw_comparison`
- `inkatoms.ml`: radial fingerprints, SOAP (dscribe), `environment_states`,
  `environment_events`
- `event_strip` defaults to a publication layout: a slab through the site, a fixed window
  (`window=`, `anchor=`), role colours for the atom and its bonded partners
- `plot_neighbours` / `neighbour_counts`: readable timelines; ML states `describe()` themselves
- Trail styles: "arrow" (smoothed route), "ghosts" (earlier copies), "line"
- Matching you control: `ids_a=` / `ids_b=`, `known={id_a: id_b}`, `pairs=`; `arrow_max=`
- Tracking without ids: `link()` (distance / velocity / learned), `fit_link_model()` (per-element
  logistic regression trained on frames with true ids), `link_accuracy()`
- Renderer split into scene.py / layers.py / render.py; per-atom `_fade`, `_layer`,
  `_scale` overrides; `vectors_front=`

## 0.2.0

- `layers=` draw order by element; `use_collection_faces()` fixes bold Helvetica on
  macOS with matplotlib < 3.11; journal widths checked against publisher guidance
  (ACS single column corrected to 3.33 in; unverified Science/Wiley entries removed)
- `depth_cue=False` switch (and `--no-depth-cue`), `fade_distance=` for a fixed fade
  length, per-element outline colours (`edge_colors=`), explicit bond lists (`bonds=[(a, b), ...]`)
- Colour atoms by a value (`color_by=`) with `colorbar()`; per-atom arrows (`vectors=`)
- Coordination polyhedra, filled rings with aromatic circles, hydrogen-bond contacts,
  bond styles (breaking / forming / dashed / hidden), bond order 1.5 (solid + dashed)
- Constraint crosses, vacancies, ghost periodic images, VESTA-style boundary atoms,
  `select.make_whole`, lattice planes, cell vectors, axis triad
- Perspective camera, fake ambient occlusion (`occlusion=`), volumetric slices from cube files
- Inputs: RDKit molecules and SMILES (with exact bond orders), MDAnalysis, pymatgen
  partial occupancies (drawn as pies)
- Presets `vesta`, `colourblind`, `dark`; journal column sizes (`figsize`)
- Labels with leader lines and repel, `measure()` for distances/angles/dihedrals
- `inset()`, `panels()`, `animate()`, interactive `picker()`, editable SVG (`editable=True`),
  automatic rasterising of very large drawings
- `inkatoms` command-line tool; documentation site with gallery; image regression tests
- Faster renderer (raw paths), metal-metal cutoffs tightened in compounds

## 0.1.0

- Depth-cued ball-and-stick drawing, Jmol/CPK/greyscale/black-and-white presets,
  highlights, selections that keep ids, periodic half-bonds, unit cell, legend,
  scale bar, bond orders, fixed inches-per-Å scale
