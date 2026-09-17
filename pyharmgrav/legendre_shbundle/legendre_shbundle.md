# SHBundle Legendre Python interface

This native Python module wraps the same C calculation routines used by
`Legendre_mex.c` and returns NumPy arrays.

## Interface

```python
from pyharmgrav.legendre_shbundle import assoc_legendre

P = assoc_legendre(degree, theta)
P, dP = assoc_legendre(degree, theta, order=5, derivatives=1)
P, dP, ddP = assoc_legendre(degree, theta, order=5, derivatives=2)
```

The full signature is:

```python
assoc_legendre(
    degree,
    theta,
    order=None,
    method="plm",
    speed_od=False,
    derivatives=0,
)
```

- `degree`: scalar or array-like of degrees. When `speed_od=True`, pass a
  scalar maximum degree/order `Lmax`.
- `theta`: scalar or array-like of co-latitudes in radians.
- `order`: associated order in PLM mode; defaults to zero and is clamped to
  `[0, Lmax]`, like the MEX function.
- `method`: `"plm"` for ordinary computation or `"xnum"` for X-number-
  stabilized computation. This choice is independent of the output mode.
- `speed_od`: if true, use the optimized calculation of all degree/order
  pairs through `Lmax`; otherwise use PLM mode.
- `derivatives`: `0`, `1`, or `2`. This is the highest derivative that is
  actually computed and controls the return value.

All results have NumPy `float64` dtype and are Fortran-contiguous so their
shape and in-memory ordering match MATLAB:

- PLM mode: `(len(theta), len(degree))`.
- `speed_od=True`: `((Lmax + 1) * (Lmax + 2) // 2, len(theta))`. Rows are ordered
  `00, 10, 20, ..., 11, 21, ..., 22, ...`, matching `assoc_legendre_mex`.

Examples:

```python
import numpy as np
from pyharmgrav.legendre_shbundle import assoc_legendre

theta = np.deg2rad([20.0, 25.0])
P = assoc_legendre(np.arange(11), theta)

P, dP = assoc_legendre(np.arange(11), theta, order=5, derivatives=1)

stable_P = assoc_legendre(np.arange(2001), theta, method="xnum")

all_P, all_dP = assoc_legendre(
    200, theta, method="plm", speed_od=True, derivatives=1
)

stable_all_P = assoc_legendre(2000, theta, method="xnum", speed_od=True)
```
