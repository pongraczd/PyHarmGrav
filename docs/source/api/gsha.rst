Spherical harmonic analysis (``gsha``)
=======================================

The :mod:`pyharmgrav.gsha` submodule provides tools for constructing layered
Earth models, performing spherical harmonic analysis, and computing isostatic
models.

Model construction
------------------

.. autofunction:: pyharmgrav.gsha.build_Model_from_files

.. autofunction:: pyharmgrav.gsha.build_Model_from_arrays

Spherical harmonic analysis
---------------------------

.. autofunction:: pyharmgrav.gsha.layer_SH_analysis

.. autofunction:: pyharmgrav.gsha.model_SH_analysis

Layer and isostatic utilities
-----------------------------

.. autofunction:: pyharmgrav.gsha.compute_equivalent_topography

.. autofunction:: pyharmgrav.gsha.compute_Pratt_compensation

.. autofunction:: pyharmgrav.gsha.merge_crust_layers

.. autofunction:: pyharmgrav.gsha.Airy_iso_surf
