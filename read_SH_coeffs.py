import pyharm as ph
import numpy as np
import os

def read_bhsc(file_path,GM=1,R=1,nmax=None):
    """
    Read spherical harmonic coefficients from a binary .bhsc file used by Curtin University.
    The data is stored as double and big endian. The file structure is:
    n_min n_max   (minimum and maximum harmonic degree)
    C-coefficients ascending in degree, followed by the order,
    S-coefficients ascending in degree, followed by the order,
    Example for a degree-10800 file:
    0 10800
    C(0,0), C(1,0), C(1,1), C(2,0), C(2,1), ... C(10800,10799), C(10800,10800),
    S(0,0), S(1,0), S(1,1), S(2,0), S(2,1), ... S(10800,10799), S(10800,10800).
    Reimplementation of read_SHCs_bshc2tri.m Matlab script at https://ddfe.curtin.edu.au/models/Earth2014/software 
    for Python, output is a pyharm.shc.Shc object instead of triangle matrix
    Default GM and R values are 1, for topography synthesis.
    ----------
    Parameters
    ----------
    file_path : str             Path to the .bhsc file.
    GM : float, optional        Gravitational parameter (default is 1 - for topography).
    R : float, optional         Reference radius (default is 1 - for topography).
    nmax : int, optional        Maximum degree and order (default is None).
    Returns
    -------
    SHCs : pyharm.shc.Shc       Spherical harmonic coefficients object.
    """
    coeffs_data = np.fromfile(file_path,dtype=np.float64)
    input_nmin = int(coeffs_data[0])
    input_nmax = int(coeffs_data[1])
    # work out number of SHC records for C and S
    n_down = int(((input_nmin-1)+1)*((input_nmin-1)+2)/2)
    n_up = int((input_nmax+1)*( input_nmax+2)/2)
    n_rows = n_up-n_down
    offset = 2  # for correct computation of indices to access C, S blocks
    # access the C and S data blocks
    # C and S are ordered by rows (n)
    C_r = coeffs_data[offset:int(offset+n_rows)]
    S_r = coeffs_data[int(offset+n_rows):int(offset+2*n_rows)]
    # C and S need to be ordered by columns (m)
    
    if nmax is not None:
        input_nmax = nmax # reset input_nmax to nmax for output SHC
        n_trunctuate = int((input_nmax+1)*( input_nmax+2)/2) - n_down
        C_r = C_r[0:n_trunctuate]
        S_r = S_r[0:n_trunctuate]

    #preallocate matrices for coefficients
    C_matrix = np.zeros((input_nmax+1, input_nmax+1))
    S_matrix = np.zeros((input_nmax+1, input_nmax+1))
    # get row and column indices for lower triangle
    rows_mat , cols_mat = np.tril_indices(input_nmax+1)
    # apply condition to only keep rows and columns within nmin and nmax
    cond = (rows_mat >= input_nmin) & (rows_mat <= input_nmax) & (cols_mat <= input_nmax)
    rows_mat = rows_mat[cond]
    cols_mat = cols_mat[cond]

    # get coefficients into matrix form
    C_matrix[rows_mat , cols_mat] = C_r
    S_matrix[rows_mat , cols_mat] = S_r
    # flatten to column-major order and remove upper triangle
    mask_out = np.tri(input_nmax+1,dtype=bool)
    C_out = C_matrix.T[mask_out.T]
    S_out = S_matrix.T[mask_out.T]
    SHCs = ph.shc.Shc.from_arrays(input_nmax,C_out,S_out,GM,R)
    return SHCs

def table_to_shcs(array,GM,R,nmax):
    n = (array[:,0]).astype(np.uint16)
    m = (array[:,1]).astype(np.uint16)
    Cnm_arr = (array[:,2])
    Snm_arr = (array[:,3])
    if nmax is None:
        nmax = int(n.max())
    #preallocate full matrices for coefficients
    Cnm = np.zeros((nmax+1,nmax+1))
    Snm = np.zeros((nmax+1,nmax+1))
    # drop coefficients beyond nmax
    cond = (n<=nmax)
    n = n[cond]
    m = m[cond]
    Cnm_arr = Cnm_arr[cond]
    Snm_arr = Snm_arr[cond]
    # fill matrices
    Cnm[n,m] = Cnm_arr
    Snm[n,m] = Snm_arr
    # flatten to column-major order and remove upper triangle
    mask_out = np.tri(nmax+1,dtype=bool)
    C_out = Cnm.T[mask_out.T]
    S_out = Snm.T[mask_out.T]
    SHCs = ph.shc.Shc.from_arrays(nmax,C_out,S_out,GM,R)
    return SHCs

def read_dat(file_path,GM=1,R=1,nmax=None):
    """
    Read from .dat text file used as input for GRAVSOFT programs, e.g. GEOCOL
    Input file structure:
    n    m       Cnm       Snm 
    |5d|5d|20.12e|20.12e|
    Example:
    0    0  1.000000000000e+00  0.000000000000e+00
    1    0  0.000000000000e+00  0.000000000000e+00
    1    1  0.000000000000e+00  0.000000000000e+00
    2    0 -4.841652170610e-04  0.000000000000e+00
    2    1 -3.388460757040e-10  1.463061089060e-09
    2    2  2.439347366210e-06 -1.400304299470e-06
    only coefficients, without header and error estimates
    ----------
    Parameters
    ----------
    file_path : str             Path to the .dat file.
    GM : float                  Gravitational parameter (default is 1 - for topography).
    R : float                   Reference radius (default is 1 - for topography).
    nmax : int, optional        Maximum degree and order (default is None).
    Returns
    -------
    SHCs : pyharm.shc.Shc       Spherical harmonic coefficients object.
    """
    # Read raw data from file
    raw_data = np.loadtxt(file_path)
    SHCs = table_to_shcs(raw_data,GM,R)
    return SHCs

def read_mat(file_path,GM=1,R=1,nmax=None):
    try:
        import h5py
    except ImportError:
        raise ImportError('h5py library needed to handle newer .mat files')
    try:
        with h5py.File(file_path, 'r') as file:
            keys_list = list(file.keys())
            assert len(keys_list) ==1 , 'Only 1 variable should be present in .mat file'
            varname = keys_list[0]
            data = np.array(file[varname]).T    # transpose because internally strored in column-major order
    except OSError:
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError('scipy library needed to handle older .mat files')
        raw_data = loadmat(file_path)
        key_list = list(raw_data.keys())
        key_list = [key for key in key_list if key[:2]!='__']
        assert len(key_list) ==1 , 'Only 1 variable should be present in .mat file'
        varname = key_list[0]
        data = np.array(raw_data[varname])
    SHCs = table_to_shcs(data,GM,R,nmax)
    return SHCs