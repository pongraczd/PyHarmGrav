from .legendre_shbundle import assoc_legendre
import pyharm as ph
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits

def _check_sorted(arr):
    """
    Check if an array is sorted in ascending or descending order.
    Raises a ValueError if the array is not sorted.
    """
    is_sorted = lambda a: (np.all(a[:-1] <= a[1:]) or np.all(a[:-1] >= a[1:]))
    if is_sorted(arr) == False:
        raise ValueError('array is not sorted')

def gshs_point(pnt : ph.crd.PointSctr | ph.crd.PointGrid |None,shcs : ph.shc.Shc,nmax : int, error : bool = False,lat : np.ndarray | None = None,
               lon : np.ndarray | None = None,r : np.ndarray | float | None = None, ptype : str | None = None,
                quantity: str = 'potential',n_workers : int = 1) -> np.ndarray:
    """Evaluate a spherical-harmonic model at scattered points or a grid.
    Spherical coordinates are expected in radians for latitude and longitude, and m for radius.
    Coordinates may be supplied either as pyharm.crd.PointSctr or pyharm.crd.PointGrid instances,
    or as separate arrays. If a point container is provided, the ``lat``, ``lon``, ``r``, and ``ptype``
    arguments must be omitted. If a point container is not provided, the coordinates must be supplied explicitly.

    :param pnt: Point container to evaluate. If provided, ``lat``, ``lon``,
        ``r``, and ``ptype`` must be omitted. If ``None``, coordinates must be
        supplied explicitly.
    :type pnt: pyharm.crd.PointSctr or pyharm.crd.PointGrid or None
    :param shcs: Spherical-harmonic coefficient model.
    :type shcs: pyharm.shc.Shc
    :param nmax: Maximum spherical-harmonic degree to use.
    :type nmax: int
    :param error: If ``True``, return the propagated coefficient-error
        magnitude instead of the potential or field value.
    :type error: bool
    :param lat: Latitudes in radians when ``pnt`` is ``None``.
    :type lat: numpy.ndarray or None
    :param lon: Longitudes in radians when ``pnt`` is ``None``.
    :type lon: numpy.ndarray or None
    :param r: Radial coordinates in the model's units when ``pnt`` is
        ``None``. For grid input, this may be a scalar or an array with the
        same shape as ``lat``.
    :type r: numpy.ndarray or float or None
    :param ptype: Input layout when ``pnt`` is ``None``. Use ``'sctr'`` or
        ``'scattered'`` for scattered points, or ``'grid'`` for a grid.
        Defaults to ``'sctr'`` when omitted.
    :type ptype: str or None
    :param n_workers: Number of worker threads to use for parallel
        computation.
    :type n_workers: int
    :param quantity: The physical quantity to evaluate. Must be one of
        ``'none'``, ``'potential'``, ``'gravity'``, ``'gravity_gradient'`` or ``'gravity_anomaly'``.
        This function not removes the normal field from coefficients, it must be done before calling this function.
        E.g. if quantity potential is used with original coefficients, the returned values will be the total potential, 
        if coefficients are corrected for the normal field before calling this function, the returned values will be the disturbing potential.
        Option ``'gravity_gradient'`` should only be used with coefficients that are corrected for the normal field. 
        If you want to compute gravity disturbance instead of gravity anomaly, use ``'gravity'`` with coefficients that are corrected for the normal field.
    :type quantity: str
    :return: Evaluated values. Scattered input returns a one-dimensional
        array; grid input returns an array with shape
        ``(lat.size, lon.size)``.
    :rtype: numpy.ndarray

    :raises ValueError: If mutually exclusive arguments are supplied, if
        ``pnt`` has an unsupported type, or if ``ptype`` has an unsupported
        value.
    """
    assert quantity in ('none','potential','gravity','gravity_gradient','gravity_anomaly'), 'Unsupported value for quantity'
    if isinstance(pnt,ph.crd.PointSctr):
        if (lat is not None) or (lon is not None) or (r is not None):
            raise ValueError('Cannot specify pnt and (lat,lon,r,ptype) too, they are mutually exclusive if pnt is given (lat,lon,r,ptype) must be all None')
        lat = pnt.lat
        lon = pnt.lon
        r = pnt.r
        f = _gshs_point_sctr(lat,lon,r,shcs,nmax,error,quantity,n_workers=n_workers)
    elif isinstance(pnt,ph.crd.PointGrid):
        if (lat is not None) or (lon is not None) or (r is not None):
            raise ValueError('Cannot specify pnt and (lat,lon,r,ptype) too, they are mutually exclusive if pnt is given (lat,lon,r,ptype) must be all None')
        lat = pnt.lat
        lon = pnt.lon
        r = pnt.r
        f = _gshs_point_grid(lat,lon,r,shcs,nmax,error,quantity,n_workers=n_workers)
    elif pnt is None:
        if ptype is None:
            ptype = 'sctr'
        if ptype in ('scattered','sctr'):
            f = _gshs_point_sctr(lat,lon,r,shcs,nmax,error,quantity,n_workers=n_workers)
        elif ptype == 'grid':
            f = _gshs_point_grid(lat,lon,r,shcs,nmax,error,quantity,n_workers=n_workers)
        else:
            raise ValueError('Unsupported value for ptype')
    else:
        raise ValueError('Unsupported input data type for pnt')
    return f

@threadpool_limits.wrap(limits={"blas": 1, "openmp": 1})
def _gshs_point_sctr(lat : np.ndarray,lon : np.ndarray,r : np.ndarray,shcs : ph.shc.Shc,nmax : int|None,error : bool,quantity : str,n_workers : int = 1) -> np.ndarray:
    if n_workers > 1:
        def worker(lat_chunk, lon_chunk, r_chunk):
            return _gshs_point_serial(lat_chunk, lon_chunk, r_chunk, shcs, nmax, error, type='sctr')
        n_workers = min(n_workers, len(lat))
        print(f"Using {n_workers} threads for parallel computation.")
        lat_chunks = np.array_split(lat, n_workers)
        lon_chunks = np.array_split(lon, n_workers)
        r_chunks = np.array_split(r, n_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = executor.map(worker, lat_chunks, lon_chunks, r_chunks)
            return np.concatenate(list(results))
    else:
        return _gshs_point_serial(lat,lon,r,shcs,nmax,error,quantity,type='sctr')

@threadpool_limits.wrap(limits={"blas": 1, "openmp": 1})
def _gshs_point_grid(lat : np.ndarray,lon : np.ndarray,r : np.ndarray,shcs : ph.shc.Shc,nmax : int|None,error : bool,quantity : str,n_workers : int = 1) -> np.ndarray:
    if n_workers > 1:
        def worker(lat_chunk, lon_chunk, r_chunk):
            return _gshs_point_serial(lat_chunk, lon_chunk, r_chunk, shcs, nmax, error, quantity, type='grid')
        n_workers = min(n_workers, len(lat))
        print(f"Using {n_workers} threads for parallel computation.")
        lat_chunks = np.array_split(lat, n_workers)
        lon_chunks = [lon for _ in range(n_workers)]
        if isinstance(r,np.ndarray):
            r_chunks = np.array_split(r, n_workers)
        elif isinstance(r,float):
            r_chunks = [r for _ in range(n_workers)]
        else:
            raise ValueError('r must be a float or an array')
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = executor.map(worker, lat_chunks, lon_chunks, r_chunks)
            return np.concatenate(list(results))
    else:
        return _gshs_point_serial(lat,lon,r,shcs,nmax,error,quantity,type='grid')

def _gshs_point_serial(lat : np.ndarray,lon : np.ndarray,r : np.ndarray|float,shcs : ph.shc.Shc,nmax : int,error : bool,quantity : str,type : str) -> np.ndarray:
    if type == 'grid':
        if isinstance(r,float):
            r = np.ones(lat.shape) * r
        elif isinstance(r,np.ndarray):
            if r.shape != lat.shape:
                raise ValueError('r must be a float or an array of the same shape as lat')
        _check_sorted(lat)
        _check_sorted(lon)
        _check_sorted(r)
    # Get the reference radius and gravitational parameter from the model.
    R = shcs.r
    GM = shcs.mu
    # Select the requested maximum degree and truncate the model if necessary.
    if nmax is None:
        nmax = shcs.nmax
    if nmax < shcs.nmax:
        shcs = ph.shc.Shcs.from_copy(shcs,nmax=nmax,nmax_shcs_out=nmax)
    elif nmax < shcs.nmax:
        raise ValueError('nmax must be greater than or equal to shcs.nmax')
    # Convert the coefficient vectors into degree/order matrices.
    C_nm_all, S_nm_all = _shcs_matrix_from_object(shcs)
    # use broadcasting instead of materializing repeated radius and degree matrices.
    # Prepare colatitudes, degrees, and radial scaling factors for all points.
    theRAD = np.pi / 2 - lat
    n = np.arange(nmax + 1)
    # accumulate each order immediately.
    # This avoids retaining four additional (number_of_points, nmax + 1) matrices for the longitude terms.
    if type == 'grid':
        field = np.zeros((len(lat), len(lon)))
    else:
        field = np.zeros(len(lat))
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)
    cos_m_lon = np.ones(len(lon))
    sin_m_lon = np.zeros(len(lon))

    if quantity == 'none':
        tk = np.ones(n.shape)
        tf = np.ones(n.shape)
        tl = n
    if quantity == 'potential':
        tk = GM / R
        tf = np.ones(n.shape)
        tl = n + 1
    elif quantity == 'gravity':
        tk = GM / R**2 * 1e5 # [mGal]
        tf = n + 1
        tl = n + 2
    elif quantity == 'gravity_anomaly':
        tk = GM / R**2 * 1e5 # [mGal]   
        tf = n - 1
        tl = n + 2
    elif quantity == 'gravity_gradient':
        tk = GM / R**3 * 1e9 # [Eotvos]
        tf = (n + 1) * (n + 2)
        tl = n + 3
    else:
        raise ValueError('Unsupported value for quantity')

    for m in range(0,nmax+1):
        # calculate only the valid triangular n >= m part.
        # Degrees below the order are identically zero.  Excluding them cuts
        # the Python/NumPy work and temporary array traffic roughly in half.
        degrees = n[m:]
        Cnm = C_nm_all[m:, m]
        Snm = S_nm_all[m:, m]
        TF = tl[None, :] * (R / r[:, None]) ** (tl[None, :])
        TFP = TF[:, m:] * assoc_legendre(degrees, theRAD, order=m)

        if error:
            TFP2 = TFP**2
            contribution_c = TFP2 @ (Cnm**2)
            contribution_s = TFP2 @ (Snm**2)
            if type == 'grid':
                field += np.outer(contribution_c, cos_m_lon**2) + np.outer(contribution_s, sin_m_lon**2)
            else:
                field += contribution_c * cos_m_lon**2 + contribution_s * sin_m_lon**2
        else:
            contribution_c = TFP @ Cnm
            contribution_s = TFP @ Snm
            if type == 'grid':
                field += np.outer(contribution_c, cos_m_lon) + np.outer(contribution_s, sin_m_lon)
            else:
                field += contribution_c * cos_m_lon + contribution_s * sin_m_lon

        # advance cos(m*lon) and sin(m*lon) by recurrence
        # instead of constructing full point-by-order trigonometric matrices.
        if m < nmax:
            next_cos_m_lon = cos_m_lon * cos_lon - sin_m_lon * sin_lon
            sin_m_lon = sin_m_lon * cos_lon + cos_m_lon * sin_lon
            cos_m_lon = next_cos_m_lon

    if error:
        field = np.sqrt(field)
    
    # Scale the field or its standard deviation by GM/R.
    return field * tk
    

def _shcs_matrix(shcs_arr,nmax):
    """Convert a 1D array of spherical harmonic coefficients into a 2D matrix.
    Input:  
        shcs_arr    : 1D numpy array of spherical harmonic coefficients of structure [C_00, C_10,C_n0,C_11,..,C_nn] or [S_00, S_10,S_n0,S_11,..,S_nn]
        nmax        : maximum degree of the spherical harmonic coefficients 
    Output:
        shcs_matrix : 2D numpy array of shape (nmax+1, nmax+1) containing the spherical harmonic coefficients
    """
    shcs_matrix = np.zeros((nmax+1,nmax+1))
    index = np.arange(0,nmax+1,1)
    n_index, m_index = np.meshgrid(index, index,indexing='ij')
    cond = n_index < m_index
    n_index[cond] = -1
    m_index[cond] = -1

    n_index = n_index.flatten(order='F')
    n_index = n_index[n_index >= 0]

    m_index = m_index.flatten(order='F')
    m_index = m_index[m_index >= 0]

    shcs_matrix[n_index,m_index] = shcs_arr
    return shcs_matrix

def _shcs_matrix_from_object(shcs):
    """Convert a pyharm.shc.Shc object into degree/order matrices for C_nm and S_nm.
    Input:
        shcs: pyharm.shc.Shc object
    Output:
        C_nm: 2D numpy array of shape (nmax+1, nmax+1) containing the C_nm coefficients
        S_nm: 2D numpy array of shape (nmax+1, nmax+1) containing the S_nm coefficients 
    """
    C_nm = _shcs_matrix(shcs.c,shcs.nmax)
    S_nm = _shcs_matrix(shcs.s,shcs.nmax)
    return C_nm, S_nm
