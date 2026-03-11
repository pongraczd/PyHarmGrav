import numpy as np 
from .pyharm_grav_shs import point_sh_synthesis, grid_sh_synthesis
import argparse
import sys

def load_config(config_file):
    params = {}
    try:
        exec(compile(open(config_file, "rb").read(), config_file, 'exec'),params)
    except: #if config file not correct, give syntax and exit
        sys.stdout.write("\nError: Cannot Load Parameters. Configuration file not given or does not exist.\n")
        sys.exit()
    params.pop('__builtins__')
    return params

def calc_grid(config):
    print('Grid synthesis')
    params = load_config(config)
    outfile = params['output_file']
    params.pop('output_file')
    result ,coords = grid_sh_synthesis(**params)
    if outfile.endswith('.nc'):
        import xarray as xr
        import rioxarray
        result_ds = xr.DataArray(result,coords,name=params['quantity'])
        result_ds.rio.write_crs(4326, inplace=True)
        result_ds.to_netcdf(outfile)
    elif outfile.endswith('.tif'):
        import xarray as xr
        import rioxarray
        result_ds = xr.DataArray(result,coords,name=params['quantity']).astype("float32")
        result_ds.rio.write_crs(4326, inplace=True)
        result_ds.rio.to_raster(outfile)
    elif outfile.endswith('.dat') or outfile.endswith('.txt'):
        lat_grid = np.repeat((coords['latitude']).reshape(-1,1),len(coords['longitude']),axis=1)
        lon_grid = np.repeat(np.expand_dims(coords['longitude'],0),len(coords['latitude']),axis=0)
        out_array = np.vstack((lat_grid.ravel(),lon_grid.ravel(),result.ravel())).T
        np.savetxt(outfile,out_array,fmt='%.8f %.8f %.12e')
    else:
        raise ValueError('Not recognised output file type')

def calc_point(config,file):
    print('Point synthesis')
    if file :
        point_numbers = True
        params = load_config(config)
        input_file = params['input_file']
        output_file = params['output_file']
        params.pop('input_file')
        params.pop('output_file')
    else:
        params = config
    if 'point_numbers' in params.keys():
        point_numbers = params['point_numbers']
        params.pop('point_numbers')
    data_in_file = np.loadtxt(input_file)
    if point_numbers:
        point_coords = data_in_file[:,1:]
    else:
        point_coords = data_in_file
    params['points'] = point_coords
    if isinstance(params['quantity'], list):
        assert len(list(set(quantity))) == len(quantity), "Duplicates in quantities"
        result = []
        for quantity in params['quantity']:
            params_local = params.copy()
            params_local['quantity'] = quantity
            result_temp = point_sh_synthesis(**params_local)
            result_temp=result_temp.reshape(-1,1)
            result.append(result_temp)
        result = np.hstack(result)
        quantity_num = len(params['quantity'])
        if 'g' in params['quantity']:
            quantity_num +=2
        if point_numbers:
            out_format = '%d %.8f %.8f %.3f ' + quantity_num * '%.12e '
        else:
            out_format = '%.8f %.8f %.3f ' + quantity_num * '%.12e '
        out_format = out_format.strip()
    else:
        result = point_sh_synthesis(**params)
        if result.ndim == 1:
            result = result.reshape(-1,1)
        if point_numbers:
            if params['quantity'] == 'g':
                out_format = '%d %.8f %.8f %.3f %.12e %.12e %.12e'
            else:
                out_format = '%d %.8f %.8f %.3f %.12e'
        else:
            out_format = '%.8f %.8f %.3f %.12e'
    output_array = np.hstack((data_in_file, result))
    np.savetxt(output_file,output_array,fmt=out_format)



def main():
    parser = argparse.ArgumentParser(description="PyHarmGrav")
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_grid = subparsers.add_parser('grid',nargs='?',help='Compute on grid')
    # config file
    parser_grid.add_argument('config',nargs='?', help='Path to config file')
    # options if no config file is used
    parser_grid.add_argument('--quantity',type=str,nargs='+')
    parser_grid.add_argument('--min_lat',type=float)
    parser_grid.add_argument('--max_lat',type=float)
    parser_grid.add_argument('--min_lon',type=float)
    parser_grid.add_argument('--max_lon',type=float)
    parser_grid.add_argument('--min_lat',type=float)
    parser_grid.add_argument('--resolution',type=float)
    parser_grid.add_argument('--shcs_data',type=str)
    parser_grid.add_argument('--resolution_unit',type=str,default='degrees')
    parser_grid.add_argument('--nmin',type=int,default=0)
    parser_grid.add_argument('--nmax',type=int)
    parser_grid.add_argument('--ellipsoid',type=str,default='GRS80')
    parser_grid.add_argument('--ref_surface_type',type=str,default='ellipsoid')
    parser_grid.add_argument('--height',type=float,default=0.0)
    parser_grid.add_argument('--GM',type=float)
    parser_grid.add_argument('--R',type=float)
    parser_grid.add_argument('--DTM_shcs_data',type=str)
    parser_grid.add_argument('--output_file',type=str)

    parser_grid.set_defaults(func=calc_grid)

    parser_point = subparsers.add_parser('point',help='Compute at scattered points')
    # config file
    parser_point.add_argument('config', help='Path to config file')
    # options if no config file is used
    parser_point.add_argument('--input_file',type=str)
    parser_point.add_argument('--shcs_data',type=str)
    parser_point.add_argument('--quantity',type=str,nargs='+')
    parser_point.add_argument('--nmin',type=int,default=0)
    parser_point.add_argument('--nmax',type=str)
    parser_point.add_argument('--ellipsoid',type=str,default='GRS80')
    parser_point.add_argument('--GM',type=float)
    parser.add_argument('--R',type=float)
    parser.add_argument('--DTM_shcs_data',type=str)
    parser.add_argument('--point_numbers',type=bool,default=True)
    parser.add_argument('--output_file',type=str)
    parser.add_argument('--normal_field_removed',default=False)

    parser_point.set_defaults(func=calc_point)
    args = parser.parse_args()
    print(args)
    args_config = args.config
    if args_config:
        args.func(args_config,True)
    else:
        args_dict = vars(args)
        args_dict.pop("config")
        args_dict.pop("command")
        args_dict.pop("func")
        args.func(args_dict,False)
    
if __name__ == '__main__':
    main()