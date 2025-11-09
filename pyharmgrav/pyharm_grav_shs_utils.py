import pyharm as ph
from .normal_grav_field import Ellipsoid
import numpy as np
from .read_SH_coeffs import read_bhsc, read_dat, read_mat
from os.path import splitext
import warnings

def geod2geoc(lla,ellipsoid):
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
    
    #ellipsoid = ellipsoid.lower().strip()
    # ensure dtype is float64
    lla = lla.astype(np.float64)
    #if ellipsoid == 'wgs84':
    #    a = 6378137 
    #    fEl = 1/298.257223563                 # Flattening of WGS84
    #    esq = fEl*(2-fEl)
    #elif ellipsoid == 'grs80':
    #    a = 6378137 
    #    esq = 0.006694380022903416 

    a = ellipsoid.a
    esq = (ellipsoid.e)**2

    D2R = np.pi/180.0
    xi = np.sqrt(1.0 - esq * np.sin(D2R*lla[:,0])*np.sin(D2R*lla[:,0]))
    p = (a / xi + lla[:,2]) * np.cos(D2R*lla[:,0])
    z = (a / xi * (1.0 - esq) + lla[:,2]) * np.sin(D2R*lla[:,0])
    r = np.sqrt(p**2+z**2)

    sph_lat = np.arctan(z/p)/D2R
    return np.concatenate((sph_lat, lla[:,1], r)).reshape(lla.shape, order='F')


def read_shcs(shcs_data,shcs_type,nmin=0,nmax=None,GM=None,R=None,ellipsoid=None):
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


def SH_synthesis(points,shcs,points_type,quantity,nmin,nmax,ellipsoid,DTM_shcs_data=None,DTM_shcs_type=None,lat_ell=None,h_ell=None,normal_field_removed = False):
    grid = True if isinstance(points,ph.crd.PointGrid) else False
    omega = ellipsoid.omega         # Earth's angular velocity in rad/s
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
            ellipsoid.subtract_normal_field(shcs,nmin)
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
            ellipsoid.subtract_normal_field(shcs,nmin)
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
                ellipsoid.subtract_normal_field(shcs,nmin)
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
                ellipsoid.subtract_normal_field(shcs,nmin)
        #T_xx = 1/r *f(1,0,0) + f(0,2,0)
        grad_xx = 1/points_r * ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=1,dlat=0,dlon=0) \
                + ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=0,dlat=2,dlon=0)
        if quantity == 'W_xx':
            grad_xx += omega**2 * np.sin(np.radians(points_lat))**2  # effect of centrifugal force
        return eotvos_scale * grad_xx
    
    elif quantity in ['T_yy', 'V_yy' , 'W_yy']:
        if quantity == 'T_yy':
            if normal_field_removed == False:
                ellipsoid.subtract_normal_field(shcs,nmin)
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
                ellipsoid.subtract_normal_field(shcs,nmin)
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
                ellipsoid.subtract_normal_field(shcs,nmin)
        #T_zz = f(2,0,0)
        grad_zz = ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=2,dlat=0,dlon=0)
        if quantity == 'W_zz':
            grad_zz +=  omega**2 * np.cos(np.radians(points_lat))**2  # effect of centrifugal force
        return eotvos_scale * grad_zz
    
    elif quantity in ['N','zeta','zeta_ell']:
        if points_type in ['spherical','sph']:
            raise ValueError('Ellipsoidal coordinates must be given!')
        if normal_field_removed == False:
            ellipsoid.subtract_normal_field(shcs,nmin)    # normal field removed from coefficients
        #if quantity == 'zeta':
            #  set heights to 0 (r to ellipsoidal radius)
            #r_ell = geod2geoc(lat_ell,longitude,np.zeros(lat_ell.length()))
            #points_ell = ph.crd.PointGrid.from_arrays(latitude, longitude, r_ell)

        T = ph.shs.point(points,shcs,nmax)
        a = ellipsoid.a
        fEl = ellipsoid.f
        esq = (ellipsoid.e)**2
        gamma_e = ellipsoid.gamma_e
        k = ellipsoid.k
        m = ellipsoid.m
        
        if grid:
            h_ell = np.repeat(h_ell.reshape(-1,1),len(points.lon),axis=1)
        gamma0 = gamma_e*(1+k*(np.sin(np.radians(lat_ell)))**2)/np.sqrt(1-esq*(np.sin(np.radians(lat_ell)))**2)
        # TODO : implement gamma0 from spherical coordinates

        if (DTM_shcs_data is None) and (quantity == 'N' or h_ell.max()<1e-10):
            raise ValueError("DTM is required for geoid undulation or height anomaly")

        if quantity == 'zeta_ell':
            fac = 1-2/a*(1+fEl+m-2*fEl*(np.sin(np.radians(lat_ell)))**2)*h_ell+3*h_ell**2/(a**2)
            gamma_h = gamma0*fac
            zeta_ell = T / gamma_h # ellipsoidal height can be nonzero, T and gamma_h refers to the actual height of point on surface
            return zeta_ell

        if quantity in ['zeta_ell','zeta']:
            zeta_ell_0 = T / gamma0 # ellipsoidal height is 0, T and gamma0 refer to ellipsoid

        
        #topo = point_sh_synthesis(points,DTM_shcs,points_type,'topo',shcs_type=DTM_shcs_type,nmax=DTM_nmax,ellipsoid=ellipsoid,GM=1,R=None)
        
        if quantity == 'N' or h_ell.max()<1e-10:
            if DTM_shcs_type is None:
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
            topo = SH_synthesis(points_topo,DTM_shcs,points_type,'topo',0,nmax,ellipsoid,grid)
            topo[topo<0] = 0
            G = 6.67259e-11
            rho = 2670  # mean density of the topography in kg/m^3
            geoid = (T-2*np.pi*G*rho*(topo**2))/gamma0 # ellipsoidal height is 0, T and gamma0 refer to ellipsoid
            h_ell = topo + geoid
        if quantity == 'N':    
            return geoid
        elif quantity == 'zeta':
            dg_dr = ph.shs.point_guru(pnt=points,shcs=shcs,nmax=nmax,dr=1,dlat=0,dlon=0) #compute delta_g
            zeta = zeta_ell_0 + dg_dr *(h_ell) / gamma0
            return zeta