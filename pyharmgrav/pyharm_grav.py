import pyharm as ph
import numpy as np
from .pyharm_grav_utils import read_shcs, geod2geoc, SH_synthesis
from os.path import splitext
import warnings
from .normal_grav_field import Ellipsoid

### FUNCTION FOR HS SYNTHESIS AT POINT
def point_sh_synthesis(points,shcs_data,points_type,quantity,shcs_type=None,nmin=0,nmax=None,ellipsoid=None,GM=None,R=None,DTM_shcs_data=None,DTM_shcs_type=None,normal_field_removed = False):
    # HANDLE DEFAULT VALUES FOR OPTIONAL PARAMETERS ------------------------------------------------------------------
    ellipsoid  = Ellipsoid(ellipsoid)
    # get shcs_type from file extension if not provided
    if shcs_type is None:
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
        if quantity in ['N','zeta']:
            max_h = 0 if points.shape[1]==2 else (points[:,2]).max()
            if max_h > 1e-6:
                if quantity == 'N':
                    raise ValueError("Reference surface to which geoid undulation is expressed is conventionally the surface \
                    of the reference ellipsoid. If you wish to compute these functionals, please set values of Ellipsoidal height to zero.")
                else:
                    warnings.warn('Heights are non-zero for computing height anomaly, they are used instead of DEM. If you want to use DEM rather, \
                            set them to zero or not specify heights at all.',UserWarning)
        lat_ell = (points[:,0]).copy()
        h_ell = np.zeros(points.shape[1]) if points.shape[1]==2 else (points[:,2]).copy()
        points = geod2geoc(points,ellipsoid)
    else:
        raise ValueError("Coordinate type not recognised")
    
    # ensure that C-contagious arrays are passed to pyharm
    latitude, longitude, radius = np.radians(np.ascontiguousarray(points[:,0])) \
    , np.radians(np.ascontiguousarray(points[:,1])), np.ascontiguousarray(points[:,2])

    if quantity == 'topo':
        radius[:] = R # r is also set to 1 in shcs for topography synthesis, so upward continuation term becomes 1

    points = ph.crd.PointSctr.from_arrays(latitude.astype(np.float64), longitude.astype(np.float64), radius.astype(np.float64))

    # SYNTHESIS OF DIFFERENT QUANTITIES -------------------------------------------------------------------------------
    # synthesis moved to separate function and handle grid setup,  synthesis function is generalized for both scatttered points and grid
    return SH_synthesis(points,shcs,points_type,quantity,nmin,nmax,ellipsoid,DTM_shcs_data,DTM_shcs_type,lat_ell,h_ell,normal_field_removed)

### FUNCTION FOR HS SYNTHESIS ON GRID
def grid_sh_synthesis(quantity,min_lat,max_lat,min_lon,max_lon,resolution,shcs_data,resolution_unit='degrees',nmin=0,nmax=None,ellipsoid=None,shcs_type=None,ref_surface_type='ellipsoid',height=0,ref_radius=None,GM=None,R=None,DTM_shcs_data=None,DTM_shcs_type=None,normal_field_removed = False):

    # HANDLE DEFAULT VALUES FOR OPTIONAL PARAMETERS ------------------------------------------------------------------
    ellipsoid  = Ellipsoid(ellipsoid)
    # get shcs_type from file extension if not provided
    if shcs_type is None:
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
    elif resolution_unit == ['s','sec','seconds']:
        lat_resolution /= 3600
        lon_resolution /= 3600
    latitudes = np.arange(max_lat,min_lat-lat_resolution/2,-1*lat_resolution) # step is negative to have latitudes in descending order
    
    # include endpoint by adding half step to max_lon (not full step to avoid floating point issues)
    longitudes = np.arange(min_lon,max_lon+lon_resolution/2,lon_resolution)
    heights = np.ones(len(latitudes))*height

    print(f"Grid size: {len(latitudes)} x {len(longitudes)} = {len(latitudes)*len(longitudes)} points")

    if ref_surface_type in ['ellipsoid','ell']:
        lat_ell = latitudes.copy()
        h_ell = heights.copy()
        lla = np.vstack((latitudes, np.zeros(len(latitudes)), heights)).T  # since latitudes are same for sphere and ellipsoid , use zero array as dummy argument
        lla = geod2geoc(lla,ellipsoid)
        latitudes = np.ascontiguousarray(lla[:,0])
        sphere_radii = np.ascontiguousarray(lla[:,2])
        points_type = 'ellipsoidal'
        
    elif ref_surface_type in ['sphere','sph']:
        ref_radius = 6378137 if ref_radius is None else ref_radius
        sphere_radii = np.ones(len(latitudes))*(ref_radius+height)
        points_type = 'spherical'
        lat_ell = None
        h_ell = None
    else:
        raise ValueError("Reference surface type not recognized")
    
    latitudes, longitudes, radius = np.radians(latitudes) \
    , np.radians(longitudes), np.ascontiguousarray(sphere_radii)

    if quantity == 'topo':
        #if R is None:
        #    shcs.rescale(mu=1,r=1)  # set GM and R to 1 for topography synthesis
        #else:
        #    shcs.rescale(mu=1)  # if R scale factor is provided, only set GM to 1
        radius[:] = R # r is also set to 1 in shcs for topography synthesis, so upward continuation term becomes 1

    if quantity in ['zeta','N','zeta_ell'] and h_ell.max()>1e-6:
        warnings.warn('height must be set to zero for computing geoid / height anomaly on a grid. Setting height to 0 ...', UserWarning)
        h_ell = np.zeros(h_ell.shape,dtype=np.float64)

    points = ph.crd.PointGrid.from_arrays(latitudes.astype(np.float64), longitudes.astype(np.float64), radius.astype(np.float64))
    if ref_surface_type in ['ellipsoid','ell']:
        coords = {'latitude': lat_ell, 'longitude': np.degrees(longitudes)}
    else:
        coords = {'latitude': np.degrees(latitudes), 'longitude': np.degrees(longitudes)}
    return SH_synthesis(points,shcs,points_type,quantity,nmin,nmax,ellipsoid,DTM_shcs_data,DTM_shcs_type,lat_ell,h_ell,normal_field_removed),  coords