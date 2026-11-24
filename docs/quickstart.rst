Quick start
===========

Install
-------

inkatoms is not on PyPI yet. Clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/JoelCrewe/InkAtoms.git
   cd InkAtoms
   pip install -e .                # numpy, scipy, matplotlib only
   pip install -e ".[all]"         # plus ASE, pymatgen, RDKit, MDAnalysis

Because the install is editable, ``git pull`` in the clone updates inkatoms.

Drawing
-------

:func:`inkatoms.draw` takes an Axes and a structure (ASE ``Atoms``, pymatgen
``Structure``/``Molecule``, RDKit ``Mol``, MDAnalysis ``Universe``, or
``(symbols, positions)``) and returns a :class:`~inkatoms.StructureArtist`,
which knows where every atom landed on the page:

.. code-block:: python

   art = ia.draw(ax, atoms, view="-y", highlight={12: "#e8503a"}, depth_fade=0.7)
   ax.annotate("site", xy=art.xy(12), xytext=(2, 3), arrowprops={"arrowstyle": "->"})

Views
-----

``view`` is the structure direction pointing out of the page: ``"z"`` is a
top view, ``"-y"`` a side view, ``"auto"`` the widest face. ``rotate="20x,-30y"``
tilts about the page axes; ``perspective=12`` sets a camera distance in Å.

Selections keep atom ids
------------------------

:mod:`inkatoms.select` crops (``slab``, ``around``, ``box``), tiles, adds
periodic images (``images``, ``boundary``) and unwraps molecules
(``make_whole``). Ids survive every step, so ``highlight=`` and ``art.xy(id)``
always mean the same atom.

Command line
------------

.. code-block:: bash

   inkatoms structure.cif -o structure.pdf --view z --polyhedra --cell --style vesta
