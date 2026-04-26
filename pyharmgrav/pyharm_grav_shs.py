import pyharm as ph
import numpy as np
from .pyharm_grav_shs_utils import read_shcs, geod2geoc, SH_synthesis, tide_system_convert
from os.path import splitext
import warnings
from .normal_grav_field import Ellipsoid
from .pyharm_grav_shs_utils import interpolate_from_raster
from numpy.typing import NDArray

### FUNCTION FOR SH SYNTHESIS AT POINT
def point_sh_synthesis(points : NDArray,shcs_data : str, points_type : str, quantity : str, nmin : int = 0, nmax : int|None = None, ellipsoid : str|list|tuple|dict|None = None, GM : float|None = None, R : float|None = None, DTM_shcs_data : str|None = None, DTM_raster : str|None = None, tide_system_conversion : list|tuple|None = None, normal_field_removed : bool = False) -> NDArray:
    """
    Compute spherical harmonic synthesis at scattered points.
    This function computes various gravity field functionals (potential, gravity, gravity gradients,
    geoid undulation, etc.) at specified points using spherical harmonic synthesis.
    Parameters
    ----------
    points : NDArray
        Input coordinates of points.
        Shape: (N, 2) for [latitude, longitude] or (N, 3) for [latitude (deg), longitude (deg), height (m)].
    shcs_data : str
        File path to spherical harmonic coefficients.
    points_type : str
        Coordinate system type of input points.
        Options: 'spherical'/'sph' or 'ellipsoidal'/'ell'.
    quantity : str
        Type of quantity to synthesize.
        Options: 
        -------------------------------------------------
        'topo'          topography [m]
        'W'             gravity potential [m^2/s^2]
        'V'             gravitational potential [m^2/s^2]
        'T'             disturbing ptential [m^2/s^2]
        'dg'            gravity anomaly [mGal]
        'dg_dist'       gravity disturbance [mGal]
        'g'             gravity vector (g_x, g_y, g_z) [m/s^2] in north-east-down local carthesian coordinate-system
        'g_abs'         gravity (scalar) [m/s^2]
        'V_xz', 'V_yz', 'V_xy', 'V_xx', 'V_yy', 'V_zz', 'V_delta'       gravitational gradients
        'W_xz', 'W_yz', 'W_xy', 'W_xx', 'W_yy', 'W_zz', 'W_delta'       gravity gradients
        'T_xz', 'T_yz', 'T_xy', 'T_xx', 'T_yy', 'T_zz', 'T_delta'       gravity anomaly gradients
        'N'             geoid undulation [m]
        'zeta'          height anomaly [m]
        'zeta_ell'      pseudo height anomaly [m]
        'xi'            deflection of vertical - north component [arcsec]
        'eta'           deflection of vertical - east component [arcsec]
        'theta'         deflection of vertical (magnitude) [arcsec]
        -------------------------------------------------
    nmin : int, optional
        Minimum degree of spherical harmonic expansion. Default: 0.
    nmax : int | None, optional
        Maximum degree of spherical harmonic expansion.
        If None, automatically determined from file. Default: None.
    ellipsoid : Ellipsoid | None, optional
        Reference ellipsoid definition. If None, GRS80 ellipsoid is used. Default: None.
    GM : float | None, optional
        Geocentric gravitational constant. Default: None.
        For 'topo' quantity defaults to 1.
    R : float | None, optional
        Reference radius (meters). Default: None.
        For 'topo' quantity defaults to 1.
    DTM_shcs_data : str | None, optional
        File path to topographic spherical harmonic coefficients. Default: None.
    DTM_raster : str | None, optional
        File path to digital terrain model raster. Default: None.
    tide_system_conversion : list or tuple | None, optional
        Two-element sequence [source_system, target_system] specifying the conversion direction.
        Valid values: 'tide-free', 'zero-tide', 'mean-tide', or None to auto-detect source. Default: None.
    normal_field_removed : bool, optional
        If True, normal field has already been removed from coefficients (default: False).
    Returns
    -------
    NDArray
        Computed field quantity at specified points. Shape depends on input:
        - Scalar quantities: 1D array of length equals number of input points
        - 'g' (gravity vector): shape (n_points, 3) in north-east-down components
    """
    # HANDLE DEFAULT VALUES FOR OPTIONAL PARAMETERS ------------------------------------------------------------------
    if ellipsoid is not None:
        ellipsoid  = Ellipsoid(ellipsoid)

    # get shcs_type from file extension if not provided
    #if shcs_type is None:
    shcs_type = splitext(shcs_data)[1][1:]  # remove the dot
    if shcs_type not in ['gfc','dat','bshc','bin','mtx','tbl','dov','mat']:   # rewrite if new format added
        raise ValueError("Not recognised file format. it must be one of these: 'gfc','dat','bshc','bin','mtx','tbl','dov','mat'  ")
    # get nmax from file if not provided and parser requires it    
    if (nmax is None) and (shcs_type in ['gfc','bin','mtx','tbl','dov']): # file types recignosed by PyHarm
        nmax = ph.shc.Shc.nmax_from_file(shcs_type,shcs_data)
    if quantity == 'topo':
        GM = GM if (GM is not None) else 1
        R = R if (R is not None) else 1
    # READ SH COEFFICIENTS FROM FILE------------------------------------------------------------------------------------

    shcs = read_shcs(shcs_data,shcs_type,nmin,nmax,GM,R,ellipsoid)

    
    if nmax is None:
        nmax = shcs.nmax

    # CONVERT TO SPHERICAL COORDIANTES IF NEEDED -----------------------------------------------------------------------

    if points_type in ['spherical','sph']:
        lat_ell = None
        h_ell = None
    elif points_type in ['ellipsoidal','ell']:
        if quantity == 'N':
            raise ValueError("Reference surface to which geoid undulation is expressed is conventionally the surface \
                    of the reference ellipsoid. If you wish to compute these functionals, please set values of Ellipsoidal height to zero.")
        elif quantity == 'zeta':
            raise ValueError("Ellipsoidal must be zero for functional 'zeta' (height anomaly). If you wish to compute height anomaly with ellipsoidal heights, please use 'zeta_ell' functional (generalised height anomaly).")
        lat_ell = (points[:,0]).copy()
        h_ell = np.zeros(points.shape[0]) if points.shape[1]==2 else (points[:,2]).copy()
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

    if quantity == 'topo':
        radius[:] = R # r is also set to 1 in shcs for topography synthesis, so upward continuation term becomes 1

    points = ph.crd.PointSctr.from_arrays(latitude.astype(np.float64), longitude.astype(np.float64), radius.astype(np.float64))

    # SYNTHESIS OF DIFFERENT QUANTITIES -------------------------------------------------------------------------------
    # synthesis moved to separate function and handle grid setup,  synthesis function is generalized for both scatttered points and grid
    result = SH_synthesis(points,shcs,points_type,quantity,nmin,nmax,ellipsoid,DTM_shcs_data,topo_heights,lat_ell,h_ell,normal_field_removed)
    if geoid_corr is not None:
        result += geoid_corr    
    return result

### FUNCTION FOR SH SYNTHESIS ON GRID
def grid_sh_synthesis(quantity : str, min_lat : float, max_lat : float, min_lon : float, max_lon : float, resolution : float|list[float]|tuple[float], shcs_data : str, resolution_unit : str = 'degrees', nmin : int = 0, nmax : int|None = None, ellipsoid : str|list|tuple|dict|None = None,ref_surface_type : str = 'ellipsoid', height : float = 0,GM : float|None = None, R : float|None = None, DTM_shcs_data : str|None =None, DTM_raster : str|None = None, tide_system_conversion : list|tuple|None = None, normal_field_removed : bool = False):
    """
    Compute spherical harmonic synthesis on a regular grid.
    This function computes various gravity field functionals (potential, gravity, gravity gradients,
    geoid undulation, etc.) at specified points using spherical harmonic synthesis.
    Parameters
    ----------
    quantity : str
        Type of quantity to synthesize.
        Options: 
        -------------------------------------------------
        'topo'          topography [m]
        'W'             gravity potential [m^2/s^2]
        'V'             gravitational potential [m^2/s^2]
        'T'             disturbing ptential [m^2/s^2]
        'dg'            gravity anomaly [mGal]
        'dg_dist'       gravity disturbance [mGal]
        'g'             gravity vector (g_x, g_y, g_z) [m/s^2] in north-east-down local carthesian coordinate-system
        'g_abs'         gravity (scalar) [m/s^2]
        'V_xz', 'V_yz', 'V_xy', 'V_xx', 'V_yy', 'V_zz', 'V_delta'       gravitational gradients
        'W_xz', 'W_yz', 'W_xy', 'W_xx', 'W_yy', 'W_zz', 'W_delta'       gravity gradients
        'T_xz', 'T_yz', 'T_xy', 'T_xx', 'T_yy', 'T_zz', 'T_delta'       gravity anomaly gradients
        'N'             geoid undulation [m]
        'zeta'          height anomaly [m]
        'zeta_ell'      pseudo height anomaly [m]
        'xi'            deflection of vertical - north component [arcsec]
        'eta'           deflection of vertical - east component [arcsec]
        'theta'         deflection of vertical (magnitude) [arcsec]
        -------------------------------------------------
    min_lat : float
        Minimum latitude of the grid in degrees.
    max_lat : float
        Maximum latitude of the grid in degrees.
    min_lon : float
        Minimum longitude of the grid in degrees.
    max_lon : float
        Maximum longitude of the grid in degrees.
    resolution : float or tuple or list
        Grid resolution. If float, applies to both latitude and longitude.
        If tuple or list of length 2, specifies [lat_resolution, lon_resolution].
    shcs_data : str
        Path to the spherical harmonic coefficients file.
    resolution_unit : str, optional
        Unit of resolution. Options: 'degrees' (default), 'm'/'min'/'minutes', 's'/'sec'/'seconds'.
    nmin : int, optional
        Minimum degree of spherical harmonic expansion (default: 0).
    nmax : int or None, optional
        Maximum degree of spherical harmonic expansion. If None, extracted from file (default: None).
    ellipsoid : str or list or tuple or dict or None, optional
        Ellipsoid specification for coordinate transformations (default: None).
    ref_surface_type : str, optional
        Reference surface type: 'ellipsoid'/'ell' or 'sphere'/'sph' (default: 'ellipsoid').
    height : float, optional
        Height above reference surface in meters (default: 0).
    GM : float | None, optional
        Geocentric gravitational constant. Default: None.
        For 'topo' quantity defaults to 1.
    R : float | None, optional
        Reference radius (meters). Default: None.
        For 'topo' quantity defaults to 1.
    DTM_shcs_data : str | None, optional
        File path to topographic spherical harmonic coefficients. Default: None.
    DTM_raster : str | None, optional
        File path to digital terrain model raster. Default: None.
    tide_system_conversion : list or tuple | None, optional
        Two-element sequence [source_system, target_system] specifying the conversion direction.
        Valid values: 'tide-free', 'zero-tide', 'mean-tide', or None to auto-detect source. Default: None.
    normal_field_removed : bool, optional
        If True, normal field has already been removed from coefficients (default: False).
    Returns
    -------
        A tuple containing:
        - Synthesis result from SH_synthesis function
        - coords : dict with 'latitude' and 'longitude' arrays in degrees
    """
    # HANDLE DEFAULT VALUES FOR OPTIONAL PARAMETERS ------------------------------------------------------------------
    if ellipsoid is not None:
        ellipsoid  = Ellipsoid(ellipsoid)
    # get shcs_type from file extension if not provided
    #if shcs_type is None:
    shcs_type = splitext(shcs_data)[1][1:]  # remove the dot
    if shcs_type not in ['gfc','dat','bshc','bin','mtx','tbl','dov','mat']:   # rewrite if new format added
        raise ValueError("Not recognised file format. it must be one of these: 'gfc','dat','bshc','bin','mtx','tbl','dov','mat' ")
    # get nmax from file if not provided and parser requires it    
    if (nmax is None) and (shcs_type in ['gfc','bin','mtx','tbl','dov']):
        nmax = ph.shc.Shc.nmax_from_file(shcs_type,shcs_data)
    if quantity == 'topo':
        GM = GM if (GM is not None) else 1
        R = R if (R is not None) else 1
    
    # READ SH COEFFICIENTS FROM FILE------------------------------------------------------------------------------------

    shcs = read_shcs(shcs_data,shcs_type,nmin,nmax,GM,R,ellipsoid)

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
    points_lon = np.repeat(np.expand_dims(longitudes,0),len(points.lat),axis=0)
    points_lat = np.repeat((latitudes).reshape(-1,1),len(points.lon),axis=1)


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

    if quantity == 'topo':
        radius[:] = R # r is also set to 1 in shcs for topography synthesis, so upward continuation term becomes 1

    if quantity in ['zeta', 'N', 'zeta_ell'] and (h_ell is not None and h_ell.max() > 1e-6):
        raise ValueError("Height must be set to zero if computing geoid undulation or height anomaly on a grid.")
    
    geoid_corr = None
    if tide_system_conversion is not None:
        lat_ell_grid = np.repeat(lat_ell.reshape(-1,1),len(points.lon),axis=1)
        geoid_corr = tide_system_conversion(shcs ,shcs_data, quantity ,tide_system_conversion ,lat_ell_grid , k = 0.3)

    points = ph.crd.PointGrid.from_arrays(latitudes.astype(np.float64), longitudes.astype(np.float64), radius.astype(np.float64))
    if ref_surface_type in ['ellipsoid','ell']:
        coords = {'latitude': lat_ell, 'longitude': np.degrees(longitudes)}
    else:
        coords = {'latitude': np.degrees(latitudes), 'longitude': np.degrees(longitudes)}
    result = SH_synthesis(points,shcs,points_type,quantity,nmin,nmax,ellipsoid,DTM_shcs_data,topo_heights,lat_ell,h_ell,normal_field_removed)
    if geoid_corr is not None:
        result += geoid_corr    
    return result,  coords