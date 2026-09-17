Introduction
============

PyHarmGrav is a Python package for spherical harmonic analysis and synthesis
of Earth's gravity field. It uses `PyHarm
<https://github.com/blazej-bucha/charm>`_ as its computational backend and
provides a Python API as well as command-line (CLI) and graphical (GUI)
interfaces.

The package supports spherical harmonic synthesis at scattered points given
in either ellipsoidal or spherical coordinates, as well as on regular grids.
Available quantities include:

*  topography [m]
* :math:`W` gravity potential, :math:`V`: gravitational potential, :math:`T`: disturbing potential [m²/s²]
* :math:`\Delta g` : gravity anomaly, :math:`\delta g`: gravity disturbance [mGal]
* :math:`\mathbf{g}`:  north-east-down gravity vector, :math:`|\mathbf{g}|`: scalar gravity [m/s²]
* :math:`V_{xz}`, :math:`V_{yz}`, :math:`V_{xy}`, :math:`V_{xx}`, :math:`V_{yy}`, :math:`V_{zz}`, and :math:`V_{\Delta}`: gravitational gradients
* :math:`W_{xz}`, :math:`W_{yz}`, :math:`W_{xy}`, :math:`W_{xx}`, :math:`W_{yy}`, :math:`W_{zz}`, and :math:`W_{\Delta}`: gravity gradients
* :math:`T_{xz}`, :math:`T_{yz}`, :math:`T_{xy}`, :math:`T_{xx}`, :math:`T_{yy}`, :math:`T_{zz}`, and :math:`T_{\Delta}`: gravity anomaly gradients
* :math:`N` : geoid undulation, :math:`\zeta` : height anomaly, :math:`\zeta_{\mathrm{ell}}` : generalized height anomaly [m]
* :math:`\xi`, :math:`\eta`, and :math:`\theta`: the north component, east component, and magnitude of the vertical deflection [arcsec].

PyHarmGrav also supports computing the error of a quantity from coefficient
errors supplied in a GFC file. Only GFC files are currently supported for this
calculation. These files contain coefficient standard deviations but not
covariances, so the calculation uses variances rather than a full covariance
matrix.

Error computation uses a less optimized backend than CHarm/PyHarm. It is based
on `SHBundle
<https://www.gis.uni-stuttgart.de/en/research/downloads/shbundle/>`_, a MATLAB
package whose associated Legendre functions are implemented in C. PyHarmGrav
reimplements SHBundle's MATLAB wrapper in Python.

PyHarmGrav can work with multiple reference ellipsoids when computing the
normal gravity field. It also includes coordinate transformations,
tide-system conversion, topographic-data handling, and spherical harmonic
analysis utilities for modelling Earth's crust.
