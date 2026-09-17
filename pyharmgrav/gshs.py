import pyharm as ph
import numpy as np
from .gshs_utils import read_shcs, geod2geoc, SH_synthesis, tide_system_convert
from os.path import splitext
from .normal_grav_field import Ellipsoid
from .gshs_utils import interpolate_from_raster
from numpy.typing import NDArray

### FUNCTION FOR SH SYNTHESIS AT POINT
def point_sh_synthesis(points : NDArray,shcs_data : str | ph.shc.Shc , points_type : str, quantity : str, nmin : int = 0, nmax : int|None = None, ellipsoid : str|list|tuple|dict|None = None, GM : float|None = None, R : float|None = None, DTM_shcs_data : str|None = None, DTM_raster : str|None = None, tide_system_conversion : list|tuple|None = None, normal_field_removed : bool = False) -> NDArray:
    """Compute spherical harmonic synthesis at scattered points.

    Evaluate gravity-field functionals such as potential, gravity, gradients,
    and geoid undulation at arbitrary locations.

    :param points: Evaluation points with shape ``(N, 2)`` for
        ``[latitude, longitude]`` or ``(N, 3)`` for
        ``[latitude_deg, longitude_deg, height_m]``.
    :type points: numpy.ndarray
    :param shcs_data: Spherical harmonic coefficients, either as a file path
        or as a ``pyharm.shc.Shc`` object.
    :type shcs_data: str or pyharm.shc.Shc
    :param points_type: Coordinate system of the input points. Use
        ``'spherical'`` or ``'sph'`` for spherical coordinates, and
        ``'ellipsoidal'`` or ``'ell'`` for ellipsoidal coordinates.
    :type points_type: str
    :param quantity: Gravity-field quantity to synthesize. Supported options
        are:

        * ``'topo'`` -- topography [m]
        * ``'W'`` -- gravity potential [m^2/s^2]
        * ``'V'`` -- gravitational potential [m^2/s^2]
        * ``'T'`` -- disturbing potential [m^2/s^2]
        * ``'dg'`` -- gravity anomaly [mGal]
        * ``'dg_dist'`` -- gravity disturbance [mGal]
        * ``'g'`` -- gravity vector ``(g_x, g_y, g_z)`` [m/s^2] in the
          north-east-down local Cartesian coordinate system
        * ``'g_abs'`` -- scalar gravity [m/s^2]
        * ``'V_xz'``, ``'V_yz'``, ``'V_xy'``, ``'V_xx'``, ``'V_yy'``,
          ``'V_zz'``, ``'V_delta'`` -- gravitational gradients
        * ``'W_xz'``, ``'W_yz'``, ``'W_xy'``, ``'W_xx'``, ``'W_yy'``,
          ``'W_zz'``, ``'W_delta'`` -- gravity gradients
        * ``'T_xz'``, ``'T_yz'``, ``'T_xy'``, ``'T_xx'``, ``'T_yy'``,
          ``'T_zz'``, ``'T_delta'`` -- gravity-anomaly gradients
        * ``'N'`` -- geoid undulation [m]
        * ``'zeta'`` -- height anomaly [m]
        * ``'zeta_ell'`` -- pseudo-height anomaly [m]
        * ``'xi'`` -- north component of the vertical deflection [arcsec]
        * ``'eta'`` -- east component of the vertical deflection [arcsec]
        * ``'theta'`` -- magnitude of the vertical deflection [arcsec]

    :type quantity: str
    :param nmin: Minimum degree of the spherical harmonic expansion.
    :type nmin: int
    :param nmax: Maximum degree of the spherical harmonic expansion. If
        ``None``, it is inferred from the coefficient data when possible.
    :type nmax: int or None
    :param ellipsoid: Reference ellipsoid definition. If ``None``, the GRS80
        ellipsoid is used.
    :type ellipsoid: Ellipsoid or str or list or tuple or dict or None
    :param GM: Geocentric gravitational constant. If ``None``, it is inferred
        from the coefficient data or the chosen ellipsoid. For ``quantity='topo'``,
        this defaults to ``1.0``.
    :type GM: float or None
    :param R: Reference radius in meters. If ``None``, it is inferred from
        the coefficient data or the chosen ellipsoid. For ``quantity='topo'``,
        this defaults to ``1.0``.
    :type R: float or None
    :param DTM_shcs_data: File path to topographic spherical harmonic
        coefficients.
    :type DTM_shcs_data: str or None
    :param DTM_raster: File path to a digital terrain model raster.
    :type DTM_raster: str or None
    :param tide_system_conversion: Two-element conversion specification
        ``[source_system, target_system]``. Valid values are ``'tide-free'``,
        ``'zero-tide'``, and ``'mean-tide'``. A ``None`` source enables
        automatic source detection.
    :type tide_system_conversion: list or tuple or None
    :param normal_field_removed: If ``True``, the normal field has already
        been removed from the coefficients.
    :type normal_field_removed: bool

    :return: Computed quantity at the input points. Scalar outputs are
        returned as 1D arrays with length equal to the number of points.
        The gravity-vector quantity ``'g'`` has shape
        ``(n_points, 3)`` in north-east-down components.
    :rtype: numpy.ndarray
    """
    # HANDLE DEFAULT VALUES FOR OPTIONAL PARAMETERS ------------------------------------------------------------------
    if ellipsoid is not None:
        ellipsoid  = Ellipsoid(ellipsoid)

    if isinstance(shcs_data, str):
        # get shcs_type from file extension
        shcs_type = splitext(shcs_data)[1][1:]  # remove the dot
        if shcs_type not in ['gfc','dat','bshc','bin','mtx','tbl','dov','mat']:   # rewrite if new format added
            raise ValueError("Not recognised file format. it must be one of these: 'gfc','dat','bshc','bin','mtx','tbl','dov','mat'  ")
        # get nmax from file if not provided and parser requires it    
        if (nmax is None) and (shcs_type in ['gfc','bin','mtx','tbl','dov']): # file types recignosed by PyHarm
            nmax = ph.shc.Shc.nmax_from_file(shcs_type,shcs_data)
        if quantity =='topo':
            GM =  1.0
            R =  1.0
        # READ SH COEFFICIENTS FROM FILE------------------------------------------------------------------------------------

        if shcs_type.lower().strip() in ['gfc','bin','mtx','tbl','dov']:
            shcs = read_shcs(shcs_data,shcs_type,nmin,nmax,None,None,ellipsoid)
        else:
            shcs = read_shcs(shcs_data,shcs_type,nmin,nmax,GM,R,ellipsoid)
    elif isinstance(shcs_data,ph.shc.Shc):
        shcs = ph.shc.Shc.from_copy(shcs_data)
    else:
        raise ValueError('Invalid data  type')

    
    if nmax is None:
        nmax = shcs.nmax

    # CONVERT TO SPHERICAL COORDIANTES IF NEEDED -----------------------------------------------------------------------

    if points_type in ['spherical','sph']:
        lat_ell = None
        h_ell = None
    elif points_type in ['ellipsoidal','ell']:
        lat_ell = (points[:,0]).copy()
        h_ell = np.zeros(points.shape[0]) if points.shape[1]==2 else (points[:,2]).copy()
        if quantity in ["N", "zeta"] and not np.allclose(h_ell, 0.0):
            raise ValueError(f"Ellipsoidal height must be zero when computing {quantity!r}.")
        
        points = geod2geoc(points,ellipsoid)
    else:
        raise ValueError("Coordinate type not recognised")
    
    if (points_type in ['spherical','sph']) and (quantity in ['N','zeta']):
            raise ValueError('Ellipsoidal coordinates must be given!')
    ## TIDE SYSTEM CONVERSION ------------------------------------------------------------------------------------------

    geoid_corr = None
    if tide_system_conversion is not None:
        print(f" Tide system conversion applied: {tide_system_conversion}")
        geoid_corr = tide_system_convert(shcs ,shcs_data, quantity ,tide_system_conversion ,lat_ell , k = 0.3)

    ## DTM heights if needed -------------------------------------------------------------------------------------------
    if DTM_raster is not None and DTM_shcs_data is not None:
        raise ValueError("Both DTM_shcs_data and DTM_raster are provided. Please provide only one of these to get height information for topography synthesis.")
    ## get height from DTM raster if DTM_shcs_data not provided but DTM_raster provided
    if DTM_raster is not None:
        topo_heights = interpolate_from_raster(DTM_raster, points[:,1], points[:,0]) # note the order of arguments for interpolation is (lon, lat)
    else:
        topo_heights = None
    
    # ensure that C-contagious arrays are passed to pyharm
    latitude, longitude, radius = np.radians(np.ascontiguousarray(points[:,0])) \
    , np.radians(np.ascontiguousarray(points[:,1])), np.ascontiguousarray(points[:,2])

    if quantity in ['topo','tws','smd']:
        radius[:] = 1.0 # r is also set to 1 in shcs for topography synthesis, so upward continuation term becomes 1

    points = ph.crd.PointSctr.from_arrays(latitude.astype(np.float64), longitude.astype(np.float64), radius.astype(np.float64))

    # SYNTHESIS OF DIFFERENT QUANTITIES -------------------------------------------------------------------------------
    # synthesis moved to separate function and handle grid setup,  synthesis function is generalized for both scatttered points and grid
    result = SH_synthesis(points,shcs,points_type,quantity,nmin,nmax,ellipsoid,DTM_shcs_data,topo_heights,lat_ell,h_ell,normal_field_removed)
    if geoid_corr is not None:
        result += geoid_corr    
    return result

### FUNCTION FOR SH SYNTHESIS ON GRID
def grid_sh_synthesis(quantity : str, min_lat : float, max_lat : float, min_lon : float, max_lon : float, resolution : float|list[float]|tuple[float], shcs_data : str | ph.shc.Shc, resolution_unit : str = 'degrees', nmin : int = 0, nmax : int|None = None, ellipsoid : str|list|tuple|dict|None = None,ref_surface_type : str = 'ellipsoid', height : float = 0,GM : float|None = None, R : float|None = None, DTM_shcs_data : str|None =None, DTM_raster : str|None = None, tide_system_conversion : list|tuple|None = None, normal_field_removed : bool = False):
    """Compute spherical harmonic synthesis on a regular grid.

    Evaluate potential, gravity, gravity gradients, geoid undulation, and
    related gravity-field quantities over a latitude-longitude grid.

    :param quantity: Gravity-field quantity to synthesize. Supported options
        are:

        * ``'topo'`` -- topography [m]
        * ``'W'`` -- gravity potential [m^2/s^2]
        * ``'V'`` -- gravitational potential [m^2/s^2]
        * ``'T'`` -- disturbing potential [m^2/s^2]
        * ``'dg'`` -- gravity anomaly [mGal]
        * ``'dg_dist'`` -- gravity disturbance [mGal]
        * ``'g_abs'`` -- scalar gravity [m/s^2]
        * ``'V_xz'``, ``'V_yz'``, ``'V_xy'``, ``'V_xx'``, ``'V_yy'``,
          ``'V_zz'``, ``'V_delta'`` -- gravitational gradients
        * ``'W_xz'``, ``'W_yz'``, ``'W_xy'``, ``'W_xx'``, ``'W_yy'``,
          ``'W_zz'``, ``'W_delta'`` -- gravity gradients
        * ``'T_xz'``, ``'T_yz'``, ``'T_xy'``, ``'T_xx'``, ``'T_yy'``,
          ``'T_zz'``, ``'T_delta'`` -- gravity-anomaly gradients
        * ``'N'`` -- geoid undulation [m]
        * ``'zeta'`` -- height anomaly [m]
        * ``'zeta_ell'`` -- pseudo-height anomaly [m]
        * ``'xi'`` -- north component of the vertical deflection [arcsec]
        * ``'eta'`` -- east component of the vertical deflection [arcsec]
        * ``'theta'`` -- magnitude of the vertical deflection [arcsec]

    :type quantity: str
    :param min_lat: Minimum grid latitude in degrees.
    :type min_lat: float
    :param max_lat: Maximum grid latitude in degrees.
    :type max_lat: float
    :param min_lon: Minimum grid longitude in degrees.
    :type min_lon: float
    :param max_lon: Maximum grid longitude in degrees.
    :type max_lon: float
    :param resolution: Grid spacing. A scalar applies to both latitude and
        longitude. A two-element sequence specifies
        ``[latitude_resolution, longitude_resolution]``.
    :type resolution: float or list[float] or tuple[float]
    :param shcs_data: Spherical harmonic coefficients, either as a file path
        or as a ``pyharm.shc.Shc`` object.
    :type shcs_data: str or pyharm.shc.Shc
    :param resolution_unit: Unit of ``resolution``. Use ``'degrees'``;
        ``'m'``, ``'min'``, or ``'minutes'`` for arcminutes; or ``'s'``,
        ``'sec'``, or ``'seconds'`` for arcseconds.
    :type resolution_unit: str
    :param nmin: Minimum degree of the spherical harmonic expansion.
    :type nmin: int
    :param nmax: Maximum degree of the spherical harmonic expansion. If
        ``None``, it is inferred from the coefficient data when possible.
    :type nmax: int or None
    :param ellipsoid: Reference ellipsoid definition. If ``None``, the GRS80
        ellipsoid is used.
    :type ellipsoid: Ellipsoid or str or list or tuple or dict or None
    :param ref_surface_type: Reference surface on which to construct the grid.
        Use ``'ellipsoid'`` or ``'ell'`` for an ellipsoid, and ``'sphere'`` or
        ``'sph'`` for a sphere.
    :type ref_surface_type: str
    :param height: Height above the reference surface in meters.
    :type height: float
    :param GM: Geocentric gravitational constant. If ``None``, it is inferred
        from the coefficient data or the chosen ellipsoid. For
        ``quantity='topo'``, this defaults to ``1.0``.
    :type GM: float or None
    :param R: Reference radius in meters. If ``None``, it is inferred from
        the coefficient data or the chosen ellipsoid. For ``quantity='topo'``,
        this defaults to ``1.0``.
    :type R: float or None
    :param DTM_shcs_data: File path to topographic spherical harmonic
        coefficients.
    :type DTM_shcs_data: str or None
    :param DTM_raster: File path to a digital terrain model raster.
    :type DTM_raster: str or None
    :param tide_system_conversion: Two-element conversion specification
        ``[source_system, target_system]``. Valid values are ``'tide-free'``,
        ``'zero-tide'``, and ``'mean-tide'``. A ``None`` source enables
        automatic source detection.
    :type tide_system_conversion: list or tuple or None
    :param normal_field_removed: If ``True``, the normal field has already
        been removed from the coefficients.
    :type normal_field_removed: bool

    :return: A two-element tuple containing the synthesized values and a
        coordinate dictionary. The dictionary contains ``'latitude'`` and
        ``'longitude'`` arrays in degrees.
    :rtype: tuple[numpy.ndarray, dict[str, numpy.ndarray]]
    """
    if quantity == 'g':
        raise ValueError(
            "grid_sh_synthesis does not support the three-component 'g' vector"
        )

    # HANDLE DEFAULT VALUES FOR OPTIONAL PARAMETERS ------------------------------------------------------------------
    if ellipsoid is not None:
        ellipsoid  = Ellipsoid(ellipsoid)

    if isinstance(shcs_data, str):
        # get shcs_type from file extension
        shcs_type = splitext(shcs_data)[1][1:]  # remove the dot
        if shcs_type not in ['gfc','dat','bshc','bin','mtx','tbl','dov','mat']:   # rewrite if new format added
            raise ValueError("Not recognised file format. it must be one of these: 'gfc','dat','bshc','bin','mtx','tbl','dov','mat' ")
        # get nmax from file if not provided and parser requires it    
        if (nmax is None) and (shcs_type in ['gfc','bin','mtx','tbl','dov']):
            nmax = ph.shc.Shc.nmax_from_file(shcs_type,shcs_data)
        if quantity == 'topo':
            GM =  1.0
            R = 1.0
        
        # READ SH COEFFICIENTS FROM FILE------------------------------------------------------------------------------------
        if shcs_type.lower().strip() in ['gfc','bin','mtx','tbl','dov']:
            shcs = read_shcs(shcs_data,shcs_type,nmin,nmax,None,None,ellipsoid)
        else:
            shcs = read_shcs(shcs_data,shcs_type,nmin,nmax,GM,R,ellipsoid)
    elif isinstance(shcs_data,ph.shc.Shc):
        shcs = ph.shc.Shc.from_copy(shcs_data)
    else:
        raise ValueError('Invalid data  type')
    
    if nmax is None:
        nmax = shcs.nmax

    # CONVERT TO SPHERICAL COORDIANTES IF NEEDED -----------------------------------------------------------------------

    if isinstance(resolution, tuple) or isinstance(resolution, list):
        if len(resolution) != 2:
            raise ValueError("If resolution is provided as a tuple or list, it must have length 2")
        lat_resolution = resolution[0]
        lon_resolution = resolution[1]
    else:
        lat_resolution = resolution
        lon_resolution = resolution
    if resolution_unit in ['m','min','minutes']:
        lat_resolution /= 60
        lon_resolution /= 60
    elif resolution_unit in ['s','sec','seconds']:
        lat_resolution /= 3600
        lon_resolution /= 3600
    latitudes = np.arange(max_lat,min_lat-lat_resolution/2,-1*lat_resolution) # step is negative to have latitudes in descending order
    
    # include endpoint by adding half step to max_lon (not full step to avoid floating point issues)
    longitudes = np.arange(min_lon,max_lon+lon_resolution/2,lon_resolution)
    heights = np.ones(len(latitudes))*height

    print(f"Grid size: {len(latitudes)} x {len(longitudes)} = {len(latitudes)*len(longitudes)} points")


    ## get points 
    points_lon = np.repeat(np.expand_dims(longitudes,0),len(latitudes),axis=0)
    points_lat = np.repeat((latitudes).reshape(-1,1),len(longitudes),axis=1)


    if DTM_raster is not None and DTM_shcs_data is not None:
        raise ValueError("Both DTM_shcs_data and DTM_raster are provided. Please provide only one of these to get height information for topography synthesis.")
    ## get height from DTM raster if DTM_shcs_data not provided but DTM_raster provided
    if DTM_raster is not None:
        topo_heights = interpolate_from_raster(DTM_raster, points_lon.ravel(), points_lat.ravel()) # note the order of arguments for interpolation is (lon, lat)
        topo_heights = topo_heights.reshape(points_lon.shape)
    else:
        topo_heights = None

    if ref_surface_type in ['ellipsoid','ell']:
        lat_ell = latitudes.copy()
        h_ell = heights.copy()
        lla = np.vstack((latitudes, np.zeros(len(latitudes)), heights)).T  # since latitudes are same for sphere and ellipsoid , use zero array as dummy argument
        lla = geod2geoc(lla,ellipsoid)
        latitudes = np.ascontiguousarray(lla[:,0])
        sphere_radii = np.ascontiguousarray(lla[:,2])
        points_type = 'ellipsoidal'
        
    elif ref_surface_type in ['sphere','sph']:
        #ref_radius = 6378137 if ref_radius is None else ref_radius
        ref_radius = shcs.r
        sphere_radii = np.ones(len(latitudes))*(ref_radius+height)
        points_type = 'spherical'
        lat_ell = None
        h_ell = None
    else:
        raise ValueError("Reference surface type not recognized")
    
    latitudes, longitudes, radius = np.radians(latitudes) \
    , np.radians(longitudes), np.ascontiguousarray(sphere_radii)

    if quantity in ['topo','tws','smd']:
        radius[:] = 1.0 # r is also set to 1 in shcs for topography synthesis, so upward continuation term becomes 1

    if quantity in ['zeta', 'N', 'zeta_ell'] and (h_ell is not None and h_ell.max() > 1e-6):
        raise ValueError("Height must be set to zero if computing geoid undulation or height anomaly on a grid.")
    
    geoid_corr = None
    if tide_system_conversion is not None:
        lat_ell_grid = None if lat_ell is None else np.repeat(
            lat_ell.reshape(-1, 1), len(longitudes), axis=1
        )
        geoid_corr = tide_system_convert(
            shcs, shcs_data, quantity, tide_system_conversion, lat_ell_grid,
            k=0.3,
        )

    points = ph.crd.PointGrid.from_arrays(latitudes.astype(np.float64), longitudes.astype(np.float64), radius.astype(np.float64))
    if ref_surface_type in ['ellipsoid','ell']:
        coords = {'latitude': lat_ell, 'longitude': np.degrees(longitudes)}
    else:
        coords = {'latitude': np.degrees(latitudes), 'longitude': np.degrees(longitudes)}
    result = SH_synthesis(points,shcs,points_type,quantity,nmin,nmax,ellipsoid,DTM_shcs_data,topo_heights,lat_ell,h_ell,normal_field_removed)
    if geoid_corr is not None:
        result += geoid_corr    
    return result,  coords
