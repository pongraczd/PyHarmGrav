# PyHarmGrav

PyHarmGrav is a Python package for spherical harmonic analysis and synthesis
of Earth's gravity field. It uses
[PyHarm](https://github.com/blazej-bucha/charm) as its main computational backend
and provides both a Python API and command-line (CLI) and graphical (GUI)
interfaces.

The package supports spherical harmonic synthesis at scattered points given
in either ellipsoidal or spherical coordinates, as well as on regular grids.
Available quantities include:

- `topo`: topography [m];
- `W`, `V`, and `T`: gravity, gravitational, and disturbing potentials
  [m²/s²];
- `dg` and `dg_dist`: gravity anomaly and gravity disturbance [mGal];
- `g` and `g_abs`: the north-east-down gravity vector and scalar gravity
  [m/s²];
- `V_xz`, `V_yz`, `V_xy`, `V_xx`, `V_yy`, `V_zz`, and `V_delta`:
  gravitational gradients;
- `W_xz`, `W_yz`, `W_xy`, `W_xx`, `W_yy`, `W_zz`, and `W_delta`: gravity
  gradients;
- `T_xz`, `T_yz`, `T_xy`, `T_xx`, `T_yy`, `T_zz`, and `T_delta`:
  gravity anomaly gradients;
- `N`, `zeta`, and `zeta_ell`: geoid undulation, height anomaly, and
  generalized height anomaly [m]; and
- `xi`, `eta`, and `theta`: the north component, east component, and
  magnitude of the vertical deflection [arcsec].

Also supports computing error of a given quantity from errors of coefficients given 
in a GFC file (only GFC files are supported yet). GFC files only contain standard
deviations of coefficients, covariances are not included. So only variances are used
for computation, not a full covariance matrix. For computing errors a less optimized
backend is used than CHarm/PyHarm, it is based on SHBundle (https://www.gis.uni-stuttgart.de/en/research/downloads/shbundle/).
SHBundle is a Matlab package, but computing of associated Legendre functions is
implemented in C. In this library that the Matlab wrapper around it was reimplemented
in Python.

PyHarmGrav can work with multiple reference ellipsoids when computing the
normal gravity field. It also includes coordinate transformations, tide-system
conversion, topographic-data handling, and spherical harmonic analysis
utilities for modelling Earth's crust.

Documentation available at: https://pyharmgrav.readthedocs.io/en/latest/
