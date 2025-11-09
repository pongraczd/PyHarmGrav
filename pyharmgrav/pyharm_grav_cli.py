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
    config_file = config.config
    params = load_config(config_file)
    outfile = params['output_file']
    params.pop('output_file')
    result ,coords = grid_sh_synthesis(**params)
    if outfile.endswith('.nc'):
        import xarray as xr
        import rioxarray
        result_ds = xr.DataArray(result,coords,name=params['quantity'])
        result_ds.rio.write_crs(4326, inplace=True)
        result_ds.to_netcdf(outfile)
    elif outfile.endswith('.dat') or outfile.endswith('.txt'):
        lat_grid = np.repeat((coords['latitude']).reshape(-1,1),len(coords['longitude']),axis=1)
        lon_grid = np.repeat(np.expand_dims(coords['longitude'],0),len(coords['latitude']),axis=0)
        out_array = np.vstack((lat_grid.ravel(),lon_grid.ravel(),result.ravel())).T
        np.savetxt(outfile,out_array,fmt='%.8f %.8f %.12e')
    else:
        raise ValueError('Not recognised output file type')

def calc_point(config):
    point_numbers = False
    print('Point synthesis')
    config_file = config.config
    params = load_config(config_file)
    input_file = params['input_file']
    output_file = params['output_file']
    params.pop('input_file')
    params.pop('output_file')
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
        result = []
        for quantity in params['quantity']:
            params_local = params.copy()
            params_local['quantity'] = quantity
            result_temp = point_sh_synthesis(**params_local)
            result_temp=result_temp.reshape(-1,1)
            result.append(result_temp)
        result = np.hstack(result)
        quantity_num = len(params['quantity'])
        if point_numbers:
            out_format = '%d %.8f %.8f %.3f ' + quantity_num * '%.12e '
        else:
            out_format = '%.8f %.8f %.3f ' + quantity_num * '%.12e '
        out_format = out_format.strip()
    else:
        result = point_sh_synthesis(**params)
        result = result.reshape(-1,1)
        if point_numbers:
            out_format = '%d %.8f %.8f %.3f %.12e'
        else:
            out_format = '%.8f %.8f %.3f %.12e'
    output_array = np.hstack((data_in_file, result))
    np.savetxt(output_file,output_array,fmt=out_format)



def main():
    parser = argparse.ArgumentParser(description="PyHarmGrav")
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_grid = subparsers.add_parser('grid',help='Compute on grid')
    parser_grid.add_argument('config', help='Path to config file')
    parser_grid.set_defaults(func=calc_grid)

    parser_point = subparsers.add_parser('point',help='Compute on grid')
    parser_point.add_argument('config', help='Path to config file')
    parser_point.set_defaults(func=calc_point)
    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__':
    main()