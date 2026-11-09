API reference
=============

Drawing
-------

.. autofunction:: inkatoms.draw
.. autoclass:: inkatoms.StructureArtist
   :members:
.. autofunction:: inkatoms.set_scale
.. autofunction:: inkatoms.structure_figure

Structures and selection
------------------------

.. autoclass:: inkatoms.Structure
.. autofunction:: inkatoms.as_structure
.. autofunction:: inkatoms.from_smiles
.. automodule:: inkatoms.select
   :members:

Bonds
-----

.. autofunction:: inkatoms.find_bonds
.. autofunction:: inkatoms.cutoff_matrix
.. autofunction:: inkatoms.bond_orders
.. autofunction:: inkatoms.find_hbonds
.. autofunction:: inkatoms.find_rings
.. autofunction:: inkatoms.coordination

Styles
------

.. autoclass:: inkatoms.Style
.. autofunction:: inkatoms.style

Annotation and figures
----------------------

.. autofunction:: inkatoms.legend
.. autofunction:: inkatoms.colorbar
.. autofunction:: inkatoms.scale_bar
.. autofunction:: inkatoms.axes_triad
.. autofunction:: inkatoms.measure
.. autofunction:: inkatoms.figsize
.. autofunction:: inkatoms.inset
.. autofunction:: inkatoms.panels
.. autofunction:: inkatoms.animate
.. autofunction:: inkatoms.picker

Volumetric data
---------------

.. autofunction:: inkatoms.density_slice
.. autofunction:: inkatoms.sample_plane
.. autofunction:: inkatoms.read_cube

Fonts
-----

.. autofunction:: inkatoms.use_collection_faces

Views
-----

.. autofunction:: inkatoms.view_matrix
.. autofunction:: inkatoms.auto_view
