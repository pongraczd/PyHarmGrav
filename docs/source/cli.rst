Command-line interface
======================

The ``pyharmgrav`` command provides ``point`` and ``grid`` subcommands for
spherical harmonic synthesis. Use the built-in help to see all available
arguments::

   pyharmgrav --help
   pyharmgrav point --help
   pyharmgrav grid --help

Point synthesis
---------------

The ``point`` subcommand evaluates one or more quantities at coordinates read
from a text file. For example::

   pyharmgrav point --input_file points.txt --points_type ellipsoidal \
       --shcs_data model.gfc --quantity dg T --output_file results.txt

By default, each input row contains a point number followed by latitude,
longitude, and optionally ellipsoidal height or spherical radius. Use
``--no_point_numbers`` when the first column is not a point identifier. Results
are appended to the input columns in the output text file. See
:func:`pyharmgrav.gshs.point_sh_synthesis` for the synthesis parameters and
supported quantities.

Grid synthesis
--------------

The ``grid`` subcommand evaluates one quantity on a regular grid. For example::

   pyharmgrav grid --quantity N --min_lat 45 --max_lat 49 \
       --min_lon 15 --max_lon 20 --resolution 1 \
       --resolution_unit minutes --shcs_data model.gfc \
       --output_file geoid.nc

Grid output can be written as NetCDF (``.nc``), GeoTIFF (``.tif``), or a text
table (``.dat`` or ``.txt``). The three-component ``g`` vector is available
only for point synthesis. See :func:`pyharmgrav.gshs.grid_sh_synthesis` for
details.

Configuration files
-------------------

Either subcommand can receive a configuration file instead of individual
options::

   pyharmgrav point sample_config/scattered_points_sample_config.conf
   pyharmgrav grid sample_config/grid_sample_config.conf

Configuration files contain Python variable assignments whose names match the
command-line arguments. They are executed as Python code, so only trusted
configuration files should be used. When a configuration file is supplied,
its values are used instead of command-line parameter values.
