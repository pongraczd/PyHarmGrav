import numpy as np
import pyharm as ph
from typing import SupportsFloat
import os
import copy

def check_scalar(x):
    return isinstance(x,SupportsFloat) and np.isscalar(x)
def import_layer(Model : dict ,n_layer : int):
    layer_name = f'l{n_layer}'
    layer_base = f'l{n_layer+1}'
    #scale 
    height_scale = Model['height_scale'] if 'height_scale' in list(Model.keys()) else 1
    dens_scale = Model['dens_scale'] if 'dens_scale' in list(Model.keys()) else 1
    # filenames
    fupper = Model[layer_name]['bound']
    flower = Model[layer_base]['bound']
    fdens = Model[layer_name]['dens']

    #if type(fupper) == str and type(flower) == str:
    #    U = np.loadtxt(fupper) * height_scale
    #    L = np.loadtxt(flower) * height_scale
    #elif isinstance(flower,np.ndarray) and isinstance(fupper,np.ndarray):
    #    U = fupper * height_scale
    #    L = flower * height_scale
    if (isinstance(flower,np.ndarray) or type(flower) == str) and (isinstance(fupper,np.ndarray) or type(fupper) == str):
        if isinstance(fupper,str):
            U = np.loadtxt(fupper) * height_scale
        else:
            U = fupper * height_scale
        if isinstance(flower,str):
            L = np.loadtxt(flower) * height_scale
        else:
            L = flower * height_scale
    elif check_scalar(fupper) and (isinstance(flower,str) or isinstance(flower,np.ndarray)):
        fupper = float(fupper)
        if isinstance(flower,str):
            L = np.loadtxt(flower) * height_scale
        else:
            L = flower * height_scale
        U = fupper * np.ones(L.shape) * height_scale
    elif check_scalar(flower) and (isinstance(fupper,str) or isinstance(fupper,np.ndarray)):
        flower = float(flower)
        if isinstance(fupper,str):
            U = np.loadtxt(fupper) * height_scale
        else:
            U = fupper * height_scale
        L = flower * np.ones(U.shape) * height_scale
    else:
        raise ValueError('layers cannot be read, wrong types')

    if isinstance(fdens,str):
        dens = np.loadtxt(fdens) * dens_scale
    elif check_scalar(fdens) or isinstance(fdens,np.ndarray):
        dens = fdens * np.ones(U.shape) * dens_scale
    else:
        raise ValueError('Unknown Type')

    if U.shape != L.shape or U.shape != dens.shape:
        print(f'shape of U {U.shape}')
        print(f'shape of L {L.shape}')
        print(f'shape of dens {dens.shape}')
        raise ValueError('All arrays must have same shape')
        

    return U,L,dens

def layer_SH_analysis(nmax : int,Re : float,rhoE : float,max_bin : int,fupper : np.ndarray|float,flower : np.ndarray|float,fdens : np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    shape = fupper.shape
    nlon = shape[1]
    nlat = shape[0]
    grid_res_lat = 180/nlat
    grid_res_lon = 360/nlon
    
    latmin = np.radians(np.arange(90-grid_res_lat,-90-grid_res_lat/2,-1*grid_res_lat))
    latmax = np.radians(np.arange(90,-90+grid_res_lat/2,-1*grid_res_lat))
    lonmin = np.radians(np.arange(0,360-grid_res_lon/2,grid_res_lon))
    lonmax = np.radians(np.arange(0+grid_res_lon,360+grid_res_lon/2,grid_res_lon))
    radius = np.ones(len(latmin))
    grd    = ph.crd.CellGrid.from_arrays(latmin, latmax, lonmin, lonmax, radius)

    # extract coefficients and indexes
    index = np.arange(0,nmax+1,1)
    n, m = np.meshgrid(index, index,indexing='ij')
    n[n < m] = -1

    n = n.flatten(order='F')
    m = m.flatten(order='F')
    cond = (n >= 0)
    n = n[cond]
    m = m[cond]

    C_nmT = np.zeros(n.shape)
    S_nmT = np.zeros(n.shape)
    
    for Hi in range(1,max_bin+1):
        m_layer = fdens * (fupper**Hi - flower**Hi)
        # convert data: h_eq is from latitude -180 to 180, not 0 to 360
        half = nlon // 2
        if check_scalar(m_layer):
            h_coeffs = ph.shc.Shc.from_zeros(nmax, mu=np.float64(1.0), r=np.float64(1.0))
            h_coeffs.set_coeffs(0, 0, c= m_layer/(Re**Hi))#(m_layer**Hi)
        else:
            m_layer =  np.roll(m_layer, shift=half, axis=1)
            h_coeffs = ph.sha.cell(grd,(m_layer/(Re**Hi)),nmax,ph.sha.CELL_AQ,1,1)

        match Hi:
            case 1:
                fac = 1* np.ones(n.shape)
            case 2:
                fac = (n+2)/2
            case 3:
                fac = (n+2)*(n+1)/6
            case 4:
                fac = (n+2)*(n+1)*(n)/24
            case 5:
                fac = (n+2)*(n+1)*(n)*(n-1)/120
            case 6:
                fac = (n+2)*(n+1)*(n)*(n-1)*(n-2)/720
            case 7:
                fac = (n+2)*(n+1)*(n)*(n-1)*(n-2)*(n-3)/5040
            case 8:
                fac = (n+2)*(n+1)*(n)*(n-1)*(n-2)*(n-3)*(n-4)/40320
            case _:
                raise ValueError(f"Unsupported Hi={Hi}")
        
        C_nmT += fac * h_coeffs.c    
        S_nmT += fac * h_coeffs.s 

    C_nmT = 3/(2*n+1)/rhoE * C_nmT
    S_nmT = 3/(2*n+1)/rhoE * S_nmT

    return C_nmT, S_nmT, n, m

def model_SH_analysis(Model,max_bin,geoid):
    sh_length = int((Model['nmax']+1)*( Model['nmax']+2)/2)
    C_nmT = np.zeros(sh_length)
    S_nmT = np.zeros(sh_length)
    G = 6.6743e-11
    #G = 6.673e-11
    Re3 = (Model['Re'])**3
    rhoE = 3*Model['GM']/(4*np.pi*G*Re3)

    all_layers = Model['nlayers']
    for i in range(1,all_layers+1):
        U,L,dens=import_layer(Model,i)
        if check_scalar(U):
            meanFix = U
        else:
            meanFix = U.max()
        fixRe = Model['Re']
        Re = Model['Re'] + meanFix
        U = U - meanFix
        L = L - meanFix
        C_nmT_layer , S_nmT_layer, n, m = layer_SH_analysis(Model['nmax'],Re,rhoE,max_bin,U,L,dens)
        C_nmT_layer = (Re/fixRe)**(n+3) * C_nmT_layer
        S_nmT_layer = (Re/fixRe)**(n+3) * S_nmT_layer
        C_nmT += C_nmT_layer    
        S_nmT += S_nmT_layer
    return C_nmT,S_nmT , n , m

def build_Model_from_files(input_folder,n_layers, Re : float, nmax : int|None, GM : float|None, height_scale : float|None = None, dens_scale : float|None = None,lowest_bound : float|None = None) -> dict:
    Model = {}
    for i in range(1,n_layers+1):
        Model[f'l{i}'] = {}
        Model[f'l{i}']['bound'] = os.path.join(input_folder,f'map-bd{i}')
        Model[f'l{i}']['dens'] = os.path.join(input_folder,f'map-ro{i}')
        if os.path.exists(Model[f'l{i}']['bound'])==False or os.path.exists(Model[f'l{i}']['dens'])==False:
            raise FileNotFoundError(f'Layer not found: l{i}')
        Model[f'l{i}']['dens']

    Model[f'l{n_layers+1}'] = {}
    if lowest_bound is not None:
        Model[f'l{n_layers+1}']['bound'] = lowest_bound
    else:
        if os.path.exists(os.path.join(input_folder,f'map-bd{n_layers+1}')):
            Model[f'l{n_layers+1}']['bound'] = os.path.join(input_folder,f'map-bd{n_layers+1}')
        else:
            raise FileNotFoundError(f'Layer not found: l{n_layers+1}')
    Model[f'l{n_layers+1}']['dens'] = None
    Model['nlayers'] = n_layers
    Model['Re'] = Re
    Model['nmax'] = nmax
    Model['GM'] = GM
    if height_scale is not None:
        Model['height_scale'] = height_scale
    if dens_scale is not None:
        Model['dens_scale'] = dens_scale

    return Model

def build_Model_from_arrays(boundaries : list[np.ndarray|float],densities : list[np.ndarray|float], Re : float, nmax : int|None, GM : float|None) -> dict:
    Model = {}
    n_layers = len(boundaries) - 1
    for i in range(1,len(boundaries)):
        Model[f'l{i}'] = {}
        Model[f'l{i}']['bound'] = boundaries[i-1]
        Model[f'l{i}']['dens'] = densities[i-1]

    Model[f'l{n_layers+1}'] = {}
    Model[f'l{n_layers+1}']['bound'] = boundaries[i]
    Model[f'l{n_layers+1}']['dens'] = None
    Model['nlayers'] = n_layers
    Model['Re'] = Re
    Model['nmax'] = nmax
    Model['GM'] = GM

    return Model

def compute_equivalent_topography(Model,rho_cr):
    _,lower_bound,_ = import_layer(Model,Model['nlayers'])
    deltaH_list = []
    for i in range(1,Model['nlayers']+1):
        U,L,dens=import_layer(Model,i)
        r_u = Model['Re'] + U
        r_l = Model['Re'] + L
        deltaH_temp = (dens/rho_cr*(r_u**3-r_l**3) + r_l**3)**(1/3) - r_l
        deltaH_list.append(deltaH_temp)
    h_eq = np.array(deltaH_list).sum(axis=0) + lower_bound

    return h_eq

# volume of layer at unit area
def layerVolume(Re,U, L):
    r_U = Re + U
    r_L = Re + L
    return r_U**3 - r_L**3

def compute_Pratt_compensation(Model, rho_litho=None):
    Re = Model['Re']
    # import last layer, needed separately
    U_last, L_last, dens_last = import_layer(Model, Model['nlayers'])

    if isinstance(L_last, np.ndarray):
        if (L_last.max() - L_last.min()) < 1e-3:
            D_p = abs(L_last.mean())
        else:
            raise ValueError('Last layer must be scalar or array with same values')
    else:
        D_p = abs(L_last)

    

    # sum of layer volumes and masses except last layer
    layers_mass = 0
    weight_sum = 0
    for i in range(1, Model['nlayers']):
        U, L, dens = import_layer(Model, i)
        V = layerVolume(Re,U, L)
        layers_mass += V * dens
        weight_sum += V

    # last layer volume
    V_last = layerVolume(Re,U_last, L_last)
    weight_sum += V_last

    # get a priori density for last layer
    if isinstance(dens_last, np.ndarray):
        rho_last = dens_last.mean() if (dens_last.max()-dens_last.min()) < 1e-3 else (_ for _ in ()).throw(ValueError('Last layer density must be scalar or array with same values'))
    else:
        rho_last = dens_last

    layers_mass_sum = layers_mass + rho_last * V_last

    # compute average density of the litosphere from the model
    if rho_litho is None:
        print((layers_mass_sum / weight_sum).std())
        rho_litho = (layers_mass_sum / weight_sum).mean()
        print(f'Average litosphere density computed from model used : {rho_litho} kg/m3')

    # compute density of last layer based on Pratt isostasy
    rho_val = (layerVolume(Re,0,-1*D_p)*rho_litho - layers_mass)/(V_last)

    return rho_val

def compute_Pratt_compensation_weighted(Model : dict, rho_m0: float, sigma_m : float, sigma_litho : float, rho_litho=None):
    Re = Model['Re']
    # import last layer, needed separately
    U_last, L_last, dens_last = import_layer(Model, Model['nlayers'])

    if isinstance(L_last, np.ndarray):
        if (L_last.max() - L_last.min()) < 1e-3:
            D_p = abs(L_last.mean())
        else:
            raise ValueError('Last layer must be scalar or array with same values')
    else:
        D_p = abs(L_last)

    

    # sum of layer volumes and masses except last layer
    layers_mass = 0
    weight_sum = 0
    for i in range(1, Model['nlayers']):
        U, L, dens = import_layer(Model, i)
        V = layerVolume(Re,U, L)
        layers_mass += V * dens
        weight_sum += V

    # last layer volume
    V_m = layerVolume(Re,U_last, L_last)
    weight_sum += V_m

    # get a priori density for last layer
    if isinstance(dens_last, np.ndarray):
        rho_last = dens_last.mean() if (dens_last.max()-dens_last.min()) < 1e-3 else (_ for _ in ()).throw(ValueError('Last layer density must be scalar or array with same values'))
    else:
        rho_last = dens_last

    layers_mass_sum = layers_mass + rho_last * V_m

    # compute average density of the litosphere from the model
    if rho_litho is None:
        print((layers_mass_sum / weight_sum).std())
        rho_litho = (layers_mass_sum / weight_sum).mean()
        print(f'Average litosphere density computed from model used : {rho_litho} kg/m3')

    # compute density of last layer based on Pratt isostasy
    V_litho = layerVolume(Re,0,-1*D_p)
    rho_val = (rho_m0 * (V_litho**2) * (sigma_litho**2)  +  V_m * (sigma_m**2) * (V_litho * rho_litho - layers_mass))/(V_m**2 * (sigma_m**2) + (V_litho**2) * (sigma_litho)**2)

    return rho_val

def merge_crust_layers(Model : dict,crustLayersMinMax : list[int]):
    m = 0
    vol = 0
    Re = Model['Re']
    Model_merged = copy.deepcopy(Model)
    assert len(crustLayersMinMax) == 2, "crustLayersMinMax must contain 2 values"
    assert crustLayersMinMax[1] > crustLayersMinMax[0], "values must contain min / max in this order"
    crustLayers = range(crustLayersMinMax[0],crustLayersMinMax[1]+1)
    for i in crustLayers:
        U,L,dens = import_layer(Model,i)
        m += layerVolume(Re,U, L)*dens
        vol += layerVolume(Re,U, L)
    for i in range(min(crustLayers)+3,Model['nlayers']+2):
        Model_merged.pop(f'l{i}')
    nlayers_new = crustLayersMinMax[0] + 1
    mantle_lyr_new = f"l{min(crustLayers) + 1}"
    last_cr_lyr_str = f"l{min(crustLayers)}"
    Model_merged[mantle_lyr_new]['bound'] = L / Model['height_scale']
    mantle_lyr_old = f"l{Model['nlayers']}"
    bottom_lyr_old = f"l{Model['nlayers']+1}"
    bottom_lyr_new = f"l{nlayers_new+1}"
    Model_merged[mantle_lyr_new]['dens'] = Model[mantle_lyr_old]['dens']
    Model_merged[bottom_lyr_new]['bound'] = Model[bottom_lyr_old]['bound']
    Model_merged[bottom_lyr_new]['dens'] = None
    Model_merged['nlayers'] = nlayers_new
    crust_dens = m / vol
    Model_merged[last_cr_lyr_str]['dens'] = crust_dens / 1e3
    return Model_merged

def Airy_iso_surf(h_eq,Re,D,rho_cr,rho_m):
    if isinstance(h_eq,np.ndarray):
        h_eq = h_eq.astype(np.float64)
    else:
        h_eq = float(h_eq)
    delta_rho = rho_m - rho_cr
    ##t = (Re/(Re-D))**2 * rho_cr /delta_rho * h_eq  # approximate solution
    t = Re - D - np.cbrt((Re-D)**3 - rho_cr/delta_rho * ((Re+h_eq)**3 - Re**3)) # exact solution
    depth = -1*D
    iso_surf = depth - t
    return iso_surf