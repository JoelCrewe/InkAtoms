---
title: 'inkatoms: publication-quality vector drawings of atomic structures in matplotlib'
tags:
  - Python
  - matplotlib
  - chemistry
  - materials science
  - visualisation
authors:
  - name: Joel Crewe
    orcid: 0000-0000-0000-0000   # TODO: your ORCID
    affiliation: 1
affiliations:
  - name: University of Oxford, United Kingdom   # TODO: department
    index: 1
date: 6 November 2026
bibliography: paper.bib
---

<!-- DRAFT for the Journal of Open Source Software. JOSS asks for 250-1000 words,
     a statement of need, and a mention of related software. Check
     https://joss.readthedocs.io before submitting: the software must be openly
     licensed, in a public repository, with documentation, tests and some history
     of use. -->

# Summary

`inkatoms` draws atomic structures (molecules, crystals, surfaces and simulation
snapshots) directly onto `matplotlib` [@Hunter2007] axes as vector graphics. A
drawing is a single depth-sorted collection of paths, so atoms, bonds,
polyhedra, arrows and the unit cell overlap correctly, the output stays sharp
in PDF and SVG, and it can be combined with ordinary plots, annotated with
`matplotlib` text and arrows, and edited afterwards in vector-graphics
software. Depth cueing (fading and shrinking towards the back), highlighted
atoms that are never faded, and fixed physical scales (inches per ångström)
make the drawings legible at the small sizes of journal figure panels.

# Statement of need

Figures of atomic structures are usually made in dedicated viewers such as
VESTA [@Momma2011] or rendered with ray tracers such as OVITO [@Stukowski2010],
then exported as raster images and assembled with the rest of a figure by hand.
This breaks the link between the data and the figure, makes consistent styling
across panels laborious, and produces raster images inside otherwise vector
figures. Plotting functions inside Python libraries (the `plot_atoms` function
of ASE [@Larsen2017] and the 2D structure plots of pymatviz) draw atoms as
circles on an existing axes but provide no depth cueing, and few or none of
the annotation tools a publication figure needs.

`inkatoms` is written for that figure-making step. It reads structures from ASE,
pymatgen [@Ong2013], RDKit and MDAnalysis [@MichaudAgrawal2011], and provides:

- depth cueing, highlights, and selections (slabs, spheres, boxes, periodic
  tiling and images) that keep atom identities, so trajectory frames can be
  compared panel by panel;
- bonds from covalent radii [@Cordero2008] with per-pair cutoffs, periodic
  half-bonds, bond orders (guessed from bond lengths or taken from RDKit),
  hydrogen-bond contacts and styles for breaking and forming bonds;
- colouring by any per-atom quantity with a colorbar, per-atom vectors,
  coordination polyhedra, rings, lattice planes, partial occupancies,
  vacancies and fixed-atom markers;
- standard colour schemes (Jmol, CPK, VESTA, a colour-blind-safe scheme based
  on @Okabe2008, greyscale and black-and-white line art), ball sizes from van
  der Waals radii [@Alvarez2013], and flat, glossy or soft vector shading;
- figure helpers: structures inset at points of a data plot (such as an
  energy profile), trajectory strips with shared framing, animations, measured
  distances and angles, scale bars, journal column sizes and a command-line
  tool.

# Acknowledgements

J.C. acknowledges a doctoral studentship funded by EPSRC, Unilever and St Cross
College, Oxford [grant number EP/XXXXXXX/1].   <!-- TODO: grant number -->

# References
