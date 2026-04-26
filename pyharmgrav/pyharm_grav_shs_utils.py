import pyharm as ph
from .normal_grav_field import Ellipsoid
import numpy as np
from .read_SH_coeffs import read_bhsc, read_dat, read_mat
from os.path import splitext
import warnings
from numpy.typing import NDArray
import rasterio as rio
from scipy.interpolate import interpn
import re

def geod2geoc(lla: NDArray[np.float64], ellipsoid: Ellipsoid) -> NDArray[np.float64]:
    """Convert geodetic coordinates to geocentric.
    Parameters
    ----------  
    lla : ndarray
        Nx3 array of geodetic coordinates (ellipsoidal latitude and longitude 
        in degrees, height in meters above the reference ellipsoid).
        if Nx2 array, a third column with zeros added (assume zero height)
    ellipsoid : Ellipsoid object with necessary parameters
    Returns
    ------- 
    geoc : ndarray
        Nx3 array of geocentric coordinates (spherical latitude and longitude 
        in degrees, radius in meters from the center of the Earth).
    """
    if lla.shape[1] ==2:
        lla = np.hstack(( lla , np.zeros((lla.shape[0],1)) ))
    elif lla.shape[1] > 3 or lla.shape[1] < 2:
        raise ValueError('array `lla` must have 2 or 3 columns')
    

    # ensure dtype is float64
    lla = lla.astype(np.float64)

    a = ellipsoid.a
    esq = (ellipsoid.e)**2

    D2R = np.pi/180.0
    xi = np.sqrt(1.0 - esq * np.sin(D2R*lla[:,0])*np.sin(D2R*lla[:,0]))
    p = (a / xi + lla[:,2]) * np.cos(D2R*lla[:,0])
    z = (a / xi * (1.0 - esq) + lla[:,2]) * np.sin(D2R*lla[:,0])
    r = np.sqrt(p**2+z**2)

    sph_lat = np.arctan(z/p)/D2R
    return np.concatenate((sph_lat, lla[:,1], r)).reshape(lla.shape, order='F')


def read_shcs(shcs_data : str ,shcs_type : str ,nmin : int = 0,nmax : int|None = None,GM : float|None = None,R : float|None = None,ellipsoid : Ellipsoid|None = None) -> ph.shc.Shc:
    """
    Read spherical harmonic coefficients (SHCs) from various file formats.
    Parameters
    ----------
    shcs_data : str
        Path to the file containing spherical harmonic coefficients or the data itself.
    shcs_type : str
        Type of file format. Supported formats include:
        - 'gfc', 'bin', 'mtx', 'tbl', 'dov': PyHarm-recognized formats
        - 'bshc': Binary format used by Curtin University
        - 'dat': text file with columns for degree, order, Cnm, Snm
        - 'mat': MATLAB .mat file
    nmin : int, optional
        Minimum degree. Coefficients with degree less than nmin will be set to zero.
        Default is 0 (no truncation).
    nmax : int, optional
        Maximum degree to read. If None, all available coefficients are read.
        Default is None.
    GM : float, optional
        Geocentric gravitational constant. If not provided, tries to read it from file given in shcs_data,
        if it is not possible, defaults to value from GRS80 reference system. Default is None.
    R : float, optional
        Reference radius in meters. If not provided, tries to read it from file given in shcs_data,
        if it is not possible, defaults to value from GRS80 reference system. Default is None.
    ellipsoid : Ellipsoid, optional
        Ellipsoid object to extract default GM and R values from, GRS80 values are used if None. Default is None.
    Returns
    -------
    shcs : Shc
        Spherical harmonic coefficients object with degree and order up to nmax,
        with coefficients below nmin set to zero if nmin > 0.
    Raises
    ------
    ValueError
        If nmin is greater than or equal to nmax.
    Warnings
    --------
    UserWarning
        If GM or R are provided and file containing coefficients also contains them (unnecessary, user-specified values ignored). 
        If GM or R are not provided file containing coefficients not contains them (defaults used).
    """
    if shcs_type.lower().strip() in ['gfc','bin','mtx','tbl','dov']: # read from file types recognised by PyHarm
        if (GM is not None) or (R is not None):
            warnings.warn('GM and R values are unnecessary for this file type, they are ignored in this case ...',UserWarning)
        shcs = ph.shc.Shc.from_file('gfc', shcs_data, nmax)
    elif shcs_type.lower().strip() == 'bshc': # read from bshc file (binary format used by Curtin University)
        if ((GM is None) or (R is None)): # need GM and R for gravity field synthesis, get default values if not provided
            warnings.warn("GM and R not provided, using default values",UserWarning)
            ellipsoid = Ellipsoid('grs80') if ellipsoid is None else ellipsoid  # default ellipsoid
            R = ellipsoid.a if R is None else R
            GM = ellipsoid.GM if GM is None else GM

        shcs = read_bhsc(shcs_data,GM=GM,R=R,nmax=nmax)
    elif shcs_type.lower().strip() in ['dat','mat']:
        if ((GM is None) or (R is None)): # need GM and R for gravity field synthesis, get default values if not provided
            warnings.warn("GM and R not provided, using default values",UserWarning)
            ellipsoid = Ellipsoid('grs80') if ellipsoid is None else ellipsoid  # default ellipsoid
            R = ellipsoid.a if R is None else R
            GM = ellipsoid.GM if GM is None else GM
        if shcs_type.lower().strip() == 'dat':
            shcs = read_dat(shcs_data,GM=GM,R=R,nmax=nmax)
        else:
            shcs = read_mat(shcs_data,GM=GM,R=R,nmax=nmax)
    
    if nmax is None:
        nmax = shcs.nmax
    
    if nmin > 0:
        if nmin >= nmax:
            raise ValueError('nmin must be smaller than nmax!')
        index = np.arange(0,nmin,1,dtype=int)
        n_index, m_index = np.meshgrid(index, index,indexing='ij')
        cond = (n_index < m_index)
        n_index[cond] = -1
        m_index[cond] = -1
        n_index = n_index.flatten(order='F')
        m_index = m_index.flatten(order='F')
        n_index = n_index[n_index >= 0]
        m_index = m_index[m_index >= 0]

        shcs.set_coeffs(n_index,m_index,np.zeros(len(n_index),dtype=np.float64),np.zeros(len(n_index),dtype=np.float64))

    return shcs


def interpolate_from_raster(raster_file: str, x: NDArray, y: NDArray) -> NDArray:
    """Interpolate raster values at specified coordinates.
    Performs bilinear interpolation on a raster dataset at given (x, y) coordinates.
    Parameters
    ----------
    raster_file : str
        Path to the raster file to interpolate from.
    x : NDArray
        Array of x-coordinates in raster's CRS.
    y : NDArray
        Array of y-coordinates in raster's CRS.
    
    Returns
    -------
    NDArray
        Array of interpolated values at the specified coordinates, dtype float64.
    Note
    ----    
    Raster CRS must match the coordinate system of the input (x, y) coordinates.
    """
    raster_dataset = rio.open(raster_file)
    interp_data=np.zeros(len(x),dtype=np.float64)
    transfInv = ~ raster_dataset.transform
    for i in range(len(x)):
        px,py=transfInv*(x[i],y[i])
        x_f=int(np.floor(px))
        y_f=int(np.floor(py))
        n_x=np.array([x_f,x_f+1],dtype=int)
        n_y=np.array([y_f,y_f+1],dtype=int)
        win=rio.windows.Window(col_off=x_f,row_off=y_f,width=2,height=2)
        cl=np.array(raster_dataset.read(window=win)[0],dtype='float64') #  reads float32 by default, but then interpolation does not work
        val=float(interpn((n_x,n_y),cl.T,(px,py)).item())
        interp_data[i]=val
    return interp_data

def get_gfc_metadata(fname: str) -> dict:
    """Extract metadata from a GFC file header.
    Parameters    
    ----------
    fname : str
        Path to the GFC file.
    Returns
    -------
    dict
        Dictionary containing the extracted metadata.
    """
    metadata : dict = {}
    head : bool = False
    with open(fname, 'r') as f:
        for line in f:
            line : str = line.strip()
            if re.match('begin_of_head',line):
                head = True
                continue
            if re.match('end_of_head',line):
                break
            if head:
                parts : list = re.split(r'\s+', line)
                if len(parts) == 2:
                    metadata[parts[0]] = parts[1]
    return metadata

def get_tide_system_from_gfc(fname):
    """Extract the tide system information from a GFC file.
    Parameters
    ----------
    fname : str
        Path to the GFC file.
    Returns
    -------
    str
        The extracted tide system information.
    """
    return get_gfc_metadata(fname)['tide_system']


def tide_system_convert(shcs : ph.shc.Shc, shcs_data: str|None, quantity : str|None,tide_system_conversion : list|tuple ,lat_ell : NDArray|None, k: float = 0.3) -> NDArray|None:
    """
    Convert spherical harmonic coefficients or compute geoid corrections between different tide systems.
    
    This function handles conversions between three tide systems commonly used in gravity field
    modeling: tide-free, zero-tide, and mean-tide. For geoid-related quantities, it computes
    a latitude-dependent correction. For other quantities, it modifies the C_20 coefficient.
    
    Parameters
    ----------
    shcs : ph.shc.Shc
        Spherical harmonic coefficient object. Modified in-place when quantity is not geoid-related.
    shcs_data : str or None
        File path to the spherical harmonic model file (used to extract source tide system if not specified).
    quantity : str or None
        Type of quantity being converted. Geoid-related quantities: 'N', 'zeta', 'zeta_ell'.
    tide_system_conversion : list or tuple
        Two-element sequence [source_system, target_system] specifying the conversion direction.
        Valid values: 'tide-free', 'zero-tide', 'mean-tide', or None to auto-detect source.
    lat_ell : NDArray or None
        Geodetic latitude in degrees. Required when quantity is geoid-related.
    k : float, optional
        Love number (default: 0.3), used to scale corrections in tide-free conversions.
    
    Returns
    -------
    NDArray or None
        Geoid correction array if quantity is geoid-related and lat_ell is provided; otherwise None.
    """
    if not (isinstance(tide_system_conversion,list) or isinstance(tide_system_conversion,tuple)):
            raise ValueError('Tide_system_conversion must be given as list or tuple.')
    if len(tide_system_conversion) !=2:
            raise ValueError('tide_system_conversion must contai 2 items: source and target')
    source,target = tide_system_conversion
    if source is None:
        if shcs_data is None:
            raise ValueError('Source tide system cannot be determined if source is None and shcs_data is None.')
        source = get_tide_system_from_gfc(shcs_data)
    if target is None:
        raise ValueError('Target tide system must be specified if tide_system_conversion is not None')

    geoid_corr = None
    if quantity in ['N','zeta','zeta_ell'] and lat_ell is not None:
        conv_base = (-0.198) * (3/2 * np.sin(np.radians(lat_ell))**2-1/2)
        # tide-free     zero-tide
        if source == 'tide-free' and target == 'zero-tide':
            geoid_corr = conv_base * k
        elif source == 'zero-tide' and target == 'tide-free':
            geoid_corr = -1 * conv_base * k
        # zero-tide     mean-tide
        elif source == 'zero-tide' and target == 'mean-tide':
            geoid_corr = conv_base
        elif source == 'mean-tide' and target == 'zero-tide':
            geoid_corr = -1 * conv_base
        # tide-free     mean-tide
        elif source == 'tide-free' and target == 'mean-tide':   
            geoid_corr =  (1+k) * conv_base
        elif source == 'mean-tide' and target == 'tide-free':   
            geoid_corr =  -1 * (1+k) * conv_base

    else:
        C_20 = shcs.get_coeffs(n=2,m=0)[0]
        # tide-free     zero-tide
        if source == 'tide-free' and target == 'zero-tide':
            C_20 = C_20 + k* (-1.39e-8)
        elif source == 'zero-tide' and target == 'tide-free':
            C_20 = C_20 - k* (-1.39e-8)
        # zero-tide     mean-tide
        elif source == 'zero-tide' and target == 'mean-tide':
            C_20 = C_20 + (-1.39e-8)
        elif source == 'mean-tide' and target == 'zero-tide':
            C_20 = C_20 - (-1.39e-8)
        # tide-free     mean-tide
        elif source == 'tide-free' and target == 'mean-tide':   
            C_20 = C_20 + (1+k)*(-1.39e-8)
        elif source == 'mean-tide' and target == 'tide-free':   
            C_20 = C_20 - (1+k)*(-1.39e-8)
        shcs.set_coeffs(n=2,m=0,c=C_20)

    return geoid_corr


def SH_synthesis(points : ph.crd.PointGrid|ph.crd.PointSctr,shcs : ph.shc.Shc,points_type : str,quantity : str,nmin : int = 0, nmax : int|None = None,ellipsoid : Ellipsoid|None = None,DTM_shcs_data : ph.shc.Shc|None = None,topo_heights : NDArray|None = None,lat_ell : NDArray|None = None,h_ell : NDArray|None = None, normal_field_removed : bool = False) -> NDArray:
    """
    Synthesize gravitational and gravity field quantities from spherical harmonic coefficients.
    This function computes various gravity field functionals (potential, gravity, gravity gradients,
    geoid undulation, etc.) at specified points using spherical harmonic synthesis.
    NOT FOR DIRECT USE
    Used by:
        pyharmgrav.point_sh_synthesis
        pyharmgrav.grid_sh_synthesis
    Parameters
    ----------
    points : ph.crd.PointGrid | ph.crd.PointSctr
        Evaluation points as either a regular grid or scattered points.
    points_type : str
        Type of input coordinates ('spherical'/'sph' or 'ellipsoidal'/'ell').
    shcs : ph.shc.Shc
        Spherical harmonic coefficients.
    quantity : str
        Type of quantity to compute. Options include:
        'V', 'topo', 'T', 'W', 'dg', 'dg_dist', 'g', 'g_abs','V_xz', 'V_yz', 'V_xy', 'V_xx', 'V_yy', 'V_zz', 'V_delta',
        'W_xz', 'W_yz', 'W_xy', 'W_xx', 'W_yy', 'W_zz', 'W_delta', 'T_xz', 'T_yz', 'T_xy', 'T_xx', 'T_yy', 'T_zz', 'T_delta'
        'N', 'zeta', 'zeta_ell', 'xi', 'eta', 'theta'
    nmin : int, optional
        Minimum spherical harmonic degree (default: 0).
    nmax : int | None, optional
        Maximum spherical harmonic degree (default: None, uses all available coefficients).
    ellipsoid : Ellipsoid | None, optional
        Reference ellipsoid object for coordinate transformations and normal field computation (default: None).
    DTM_shcs_data : ph.shc.Shc | None, optional
        Digital Terrain Model as spherical harmonic coefficients, required for geoid undulation (default: None).
    lat_ell : NDArray | None, optional
        Ellipsoidal latitude array for height anomaly and deflection computations (default: None).
    h_ell : NDArray | None, optional
        Ellipsoidal height array for height anomaly and deflection computations (default: None).
    normal_field_removed : bool, optional
        If True, normal field has already been removed from coefficients (default: False).
    Returns
    -------
    NDArray
        Computed field quantity at specified points. Shape depends on input:
        - Scalar quantities: 1D array of length equals number of input points
        - 'g' (gravity vector): shape (n_points, 3) in north-east-down components
    """
    grid = True if isinstance(points,ph.crd.PointGrid) else False
    if ellipsoid is not None:
        omega = ellipsoid.omega         # Earth's angular velocity in rad/s
    else:
        omega = 7.292115e-05  
    eotvos_scale = 1e9              # scale factor for gravity gradients to Eötvös unit

    if normal_field_removed == True and (quantity in ['V','topo','W','g', 'g_abs','V_xz' , 'W_xz','V_yz' , 'V_zz' , 'W_zz'\
                                     'W_yz','V_xy' , 'W_xy','V_yy' , 'W_yy','V_xx' , 'W_xx', 'V_delta' , 'W_delta']) :
        raise ValueError('Without normal field, cannot compute this functional')

    # Potential unit: m^2/s^2
    if grid==False:
        points_r = points.r
        points_lat = points.lat
        #points_lon = points.lon
    else:
        points_r = np.repeat((points.r).reshape(-1,1),len(points.lon),axis=1)
        points_lat = np.repeat((points.lat).reshape(-1,1),len(points.lon),axis=1)
        if lat_ell is not None:
            lat_ell = np.repeat(lat_ell.reshape(-1,1),len(points.lon),axis=1)
        
        #points_lon = np.repeat(np.expand_dims(points.lon,0),len(points.lat),axis=0)
    if quantity in ['V','topo']:     # gravitational potential
        potential = ph.shs.point(points,shcs,nmax)
        return potential
    
    elif quantity == 'T':   # disturbing potential
        if normal_field_removed == False:
            ellipsoid.subtract_normal_field(shcs,nmin,inplace=True)
        potential = ph.shs.point(points,shcs,nmax)
        return potential
    
    elif quantity == 'W':   # gravity potential
        potential = ph.shs.point(points,shcs,nmax)                              # gravitational potential
        potential += 0.5 * omega**2 * (points_r**2) * np.cos(points_lat)**2     # effect of centrifugal force
        return potential
    
    # Gravity anomaly   unit: mGal
    elif quantity == 'dg':
        # Gravity anomaly with spherical approximation, result is in mGal
        # validated with GEOCOL
        if normal_field_removed == False:
            ellipsoid.subtract_normal_field(shcs,nmin,inplace=True)
        # extract coefficients and indexes
        index = np.arange(0,nmax+1,1)
        n_index, m_index = np.meshgrid(index, index,indexing='ij')
        n_index[n_index < m_index] = -1
        
        n_index = n_index.flatten(order='F')
        n_index = n_index[n_index >= 0]

        c_coeffs = shcs.c
        s_coeffs = shcs.s
        # multiply coefficients by (n-1) factor
        c_coeffs = c_coeffs * (n_index - 1) * 1e5
        s_coeffs = s_coeffs * (n_index - 1) * 1e5
        R = shcs.r
        mu = shcs.mu
        shcs = None
        shcs = ph.shc.Shc.from_arrays(nmax,c_coeffs,s_coeffs,mu,R)
        dg = 1/points_r * ph.shs.point(points,shcs,nmax)
        return dg
    
    elif quantity == 'dg_dist':
        if normal_field_removed == False:
            ellipsoid.subtract_normal_field(shcs,nmin)
        dg_dr = ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=1,dlat=0,dlon=0)
        return -1e5 * dg_dr
    
    # Gravity vector / gravity magnitude, unit: m/s^2
    elif quantity in ['g', 'g_abs']:
        gx,gy,gz = ph.shs.point_grad1(pnt=points,shcs=shcs,nmax=nmax)
        # convert to north-east-down system
        gy = -gy  
        gz = -gz
        if quantity == 'g_abs':
            grav = np.sqrt(gx**2 + gy**2 + gz**2)
        else:
            grav = np.hstack((gx.reshape(-1,1),gy.reshape(-1,1), gz.reshape(-1,1)))
        return grav
    
    ## Gravity gradients, unit: E
    # Horizontal gradients
    elif quantity in ['T_xz', 'V_xz' , 'W_xz']:
        if quantity == 'T_xz':
            if normal_field_removed == False:
                ellipsoid.subtract_normal_field(shcs,nmin,inplace=True)
        #T_xz = (1 / r) * f(0,1,0) - f(1,1,0)
        grad_xz = (1 / points_r) * ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=1,dlon=0) \
             - ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=1,dlat=1,dlon=0)
        if quantity == 'W_xz':
            grad_xz += 0.5*omega**2 * np.sin(np.radians(2*points_lat))  # effect of centrifugal force
        return eotvos_scale * grad_xz
    
    elif quantity in ['T_yz', 'V_yz' , 'W_yz']:
        #if quantity == 'T_yz':
        #    subtract_normal_field(shcs, ellipsoid)  -- unnecessary, since W_yz = V_yz = T_yz
        #T_yz  = (1/r)* f(0,0,1) - f(1,0,1)
        grad_yz = (1 / points_r) * ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=0,dlon=1) \
             - ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=1,dlat=0,dlon=1)
        # if quantity == 'W_yz': -- unnecessary
        return eotvos_scale * grad_yz
    
    # Curvature gradients
    elif quantity in ['T_xy', 'V_xy' , 'W_xy']:
        #if quantity == 'T_xy':
        #    subtract_normal_field(shcs, ellipsoid)  -- unnecessary, since W_xy = V_xy = T_xy
        #T_xy = f(0,1,1) + 1/r*tan(phi)*f(0,0,1)
        grad_xy = ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=1,dlon=1) \
                + 1/points_r * np.tan(np.radians(points_lat)) * ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=0,dlon=1)
        #if quantity == 'W_xy':  -- unnecessary
        return eotvos_scale * grad_xy
    
    elif quantity in ['T_xx', 'V_xx' , 'W_xx']:
        if quantity == 'T_xx':
            if normal_field_removed == False:
                ellipsoid.subtract_normal_field(shcs,nmin,inplace=True)
        #T_xx = 1/r *f(1,0,0) + f(0,2,0)
        grad_xx = 1/points_r * ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=1,dlat=0,dlon=0) \
                + ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=2,dlon=0)
        if quantity == 'W_xx':
            grad_xx += omega**2 * np.sin(np.radians(points_lat))**2  # effect of centrifugal force
        return eotvos_scale * grad_xx
    
    elif quantity in ['T_yy', 'V_yy' , 'W_yy']:
        if quantity == 'T_yy':
            if normal_field_removed == False:
                ellipsoid.subtract_normal_field(shcs,nmin,inplace=True)
        #T_yy = 1/r *f(1,0,0) + tan(phi)/r*f(0,1,0) + f(0,0,2)
        grad_yy = 1/points_r * ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=1,dlat=0,dlon=0) \
                + np.tan(np.radians(points_lat))/points_r * ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=1,dlon=0) \
                + ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=0,dlon=2)
        if quantity == 'W_yy':
            grad_yy += omega**2 * np.cos(np.radians(2*points_lat))  # effect of centrifugal force
        return eotvos_scale * grad_yy
    
    elif quantity in ['T_delta', 'V_delta' , 'W_delta']:
        if quantity == 'T_delta':
            if normal_field_removed == False:
                ellipsoid.subtract_normal_field(shcs,nmin,inplace=True)
        #T_delta =  tan(phi)/r*f(0,1,0) + f(0,0,2) - f(0,2,0)
        grad_delta = np.tan(np.radians(points_lat))/points_r*ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=1,dlon=0)\
            + ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=0,dlon=2) \
            - ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=2,dlon=0)
        if quantity == 'W_delta':
            grad_delta +=  omega**2 * (1 - 3*np.sin(np.radians(points_lat))**2)  # effect of centrifugal force
        return eotvos_scale * grad_delta

    # Vertical gradient
    elif quantity in ['T_zz', 'V_zz' , 'W_zz']:
        if quantity == 'T_zz':
            if normal_field_removed == False:
                ellipsoid.subtract_normal_field(shcs,nmin,inplace=True)
        #T_zz = f(2,0,0)
        grad_zz = ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=2,dlat=0,dlon=0)
        if quantity == 'W_zz':
            grad_zz +=  omega**2 * np.cos(np.radians(points_lat))**2  # effect of centrifugal force
        return eotvos_scale * grad_zz
    
    elif quantity in ['N','zeta','zeta_ell']:
        if normal_field_removed == False:
            if quantity == 'zeta':
                shcs_copy = ph.shc.Shc.from_copy(shcs)
            ellipsoid.subtract_normal_field(shcs,nmin,inplace=True)    # normal field removed from coefficients
        else:
            if quantity == 'zeta':
                raise ValueError('Without normal field, cannot compute zeta (height anomaly). Please set normal_field_removed to False or compute zeta_ell instead.')

        T = ph.shs.point(points,shcs,nmax)
        
        if grid and ellipsoid:
            h_ell = np.repeat(h_ell.reshape(-1,1),len(points.lon),axis=1)
        if ellipsoid is not None:
            #gamma0 = gamma_e*(1+k*(np.sin(np.radians(lat_ell)))**2)/np.sqrt(1-esq*(np.sin(np.radians(lat_ell)))**2)
            gamma0 = ellipsoid.gamma0(lat_ell)
        else:
            gamma0 = shcs.mu / shcs.r**2

        if quantity == 'zeta_ell':
            if ellipsoid is not None:
                #fac = 1-2/a*(1+fEl+m-2*fEl*(np.sin(np.radians(lat_ell)))**2)*h_ell+3*h_ell**2/(a**2)
                gamma_h = ellipsoid.gamma_h(lat_ell,h_ell)
            else:
                if h_ell is not None:
                    raise ValueError('ERROR! Spherical coordinates used but ellipsoidal heights given, too.')
                gamma_h = gamma0
            #gamma_h = gamma0*fac
            zeta_ell = T / gamma_h # ellipsoidal height can be nonzero, T and gamma_h refers to the actual height of point on surface
            return zeta_ell
        else:
            if (DTM_shcs_data is None) and (quantity == 'N' or h_ell is None or h_ell.max()<1e-10):
                raise ValueError("DTM is required for geoid undulation or height anomaly")

            if DTM_shcs_type is not None:
                DTM_shcs_type = splitext(DTM_shcs_data)[1][1:]  # remove the dot
                DTM_shcs = read_shcs(DTM_shcs_data,DTM_shcs_type,0,nmax,ellipsoid=ellipsoid,GM=1,R=1)
                radius = points.r
                radius_topo = radius.copy()
                radius_topo[:] = DTM_shcs.r # r is also set to 1 in shcs for topography synthesis, so upward continuation term becomes 1
                if grid==True:
                    points_topo = ph.crd.PointGrid.from_arrays(points.lat,points.lon,radius_topo)
                else:
                    points_topo = ph.crd.PointSctr.from_arrays(points.lat,points.lon,radius_topo)
                print('Topography synthetised with spherical harmonics')
                topo = SH_synthesis(points_topo,DTM_shcs,points_type,'topo',0,nmax,ellipsoid)
                topo[topo<0] = 0
            elif topo_heights is not None:
                topo = topo_heights
                topo[topo<0] = 0
            
            #h_ell = topo + geoid
        if quantity == 'N':    
            G = 6.67259e-11
            rho = 2670  # mean density of the topography in kg/m^3
            geoid = (T-2*np.pi*G*rho*(topo**2))/gamma0 # ellipsoidal height is 0, T and gamma0 refer to ellipsoid
            return geoid
        elif quantity == 'zeta':
            zeta_ell_0 = T / gamma0 
            h_ell = topo + zeta_ell_0
            if grid:
                dg_dr = ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=1,dlat=0,dlon=0) #compute delta_g
                zeta = zeta_ell_0 + dg_dr *(h_ell) / gamma0
            else:
                r_t = shcs.r.copy() + h_ell
                points_t = ph.crd.PointSctr.from_arrays(points.lat.copy(), points.lon.copy(), r_t)
                W_t = SH_synthesis(points_t,shcs_copy,points_type,'W',0,nmax,ellipsoid)
                rt0 = r_t - zeta_ell_0
                points_t_0 = ph.crd.PointSctr.from_arrays(points.lat.copy(), points.lon.copy(), rt0)
                U_t0 = ellipsoid.normal_potential_sph(points_t_0,shcs_copy.mu,shcs_copy.r)
                zeta = zeta_ell_0 + (W_t - U_t0) / gamma0
            return zeta

    ## deflections of vertical
    elif quantity in ['xi','eta','theta']:
        if (points_type in ['spherical','sph']):
            raise ValueError('Ellipsoidal coordinates must be given!')
        a = ellipsoid.a
        fEl = ellipsoid.f
        esq = (ellipsoid.e)**2
        gamma_e = ellipsoid.gamma_e
        k = ellipsoid.k
        m = ellipsoid.m
        if grid:
            h_ell = np.repeat(h_ell.reshape(-1,1),len(points.lon),axis=1)
        gamma0 = gamma_e*(1+k*(np.sin(np.radians(lat_ell)))**2)/np.sqrt(1-esq*(np.sin(np.radians(lat_ell)))**2)
        fac = 1-2/a*(1+fEl+m-2*fEl*(np.sin(np.radians(lat_ell)))**2)*h_ell+3*h_ell**2/(a**2)
        gamma_h = gamma0*fac 
        if normal_field_removed == False:
            ellipsoid.subtract_normal_field(shcs,nmin,inplace=True)
        rad2sec = 180/np.pi * 3600
        if quantity == 'xi':
            xi = -1/gamma_h * ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=1,dlon=0)
            return xi * rad2sec
        elif quantity == 'eta':
            eta = -1/gamma_h * ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=0,dlon=1)
            return eta * rad2sec
        elif quantity == 'theta':
            xi = -1/gamma_h * ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=1,dlon=0)
            eta = -1/gamma_h * ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=0,dlon=1)
            theta = np.sqrt(xi**2+eta**2)
            return theta * rad2sec
