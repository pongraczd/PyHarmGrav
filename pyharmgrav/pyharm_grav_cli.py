import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .gshs import grid_sh_synthesis, point_sh_synthesis


GRID_SYNTHESIS_PARAMETERS = (
    'quantity', 'min_lat', 'max_lat', 'min_lon', 'max_lon', 'resolution',
    'shcs_data', 'resolution_unit', 'nmin', 'nmax', 'ellipsoid',
    'ref_surface_type', 'height', 'GM', 'R', 'DTM_shcs_data', 'DTM_raster',
    'tide_system_conversion', 'normal_field_removed',
)
GRID_CLI_PARAMETERS = GRID_SYNTHESIS_PARAMETERS + ('output_file',)
GRID_REQUIRED_PARAMETERS = (
    'quantity', 'min_lat', 'max_lat', 'min_lon', 'max_lon', 'resolution',
    'shcs_data', 'output_file',
)

POINT_SYNTHESIS_PARAMETERS = (
    'shcs_data', 'points_type', 'quantity', 'nmin', 'nmax', 'ellipsoid',
    'GM', 'R', 'DTM_shcs_data', 'DTM_raster', 'tide_system_conversion',
    'normal_field_removed',
)
POINT_CLI_PARAMETERS = (
    'input_file', 'points_type', 'shcs_data', 'quantity', 'nmin', 'nmax',
    'ellipsoid', 'GM', 'R', 'DTM_shcs_data', 'DTM_raster', 'point_numbers',
    'tide_system_conversion', 'output_file', 'normal_field_removed',
)
POINT_REQUIRED_PARAMETERS = (
    'input_file', 'points_type', 'shcs_data', 'quantity', 'output_file',
)


def _parse_bool(value):
    """Parse common command-line boolean representations."""
    if isinstance(value, bool):
        return value
    normalized = value.casefold()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value, received {value!r}"
    )


def load_config(config_file):
    params = {}
    try:
        source = Path(config_file).read_bytes()
        exec(compile(source, str(config_file), 'exec'), params)
    except Exception as exc:
        raise ValueError(
            f"Cannot load configuration file {str(config_file)!r}: {exc}"
        ) from exc
    return {
        name: value
        for name, value in params.items()
        if name != '__builtins__'
    }


def _validate_params(params, allowed, required):
    if not isinstance(params, Mapping):
        raise TypeError('Parameters must be supplied as a mapping')

    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported parameters: {', '.join(unknown)}")

    missing = [name for name in required if params.get(name) is None]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")


def _normalize_grid_result(result, coords):
    n_lat = len(coords['latitude'])
    n_lon = len(coords['longitude'])
    result = np.asarray(result)

    if result.shape == (n_lat, n_lon):
        return result
    if result.size == n_lat * n_lon:
        return result.reshape(n_lat, n_lon)
    raise ValueError('Grid result shape does not match the coordinate arrays')


def _write_grid_output(outfile, result, coords, quantity):
    suffix = Path(outfile).suffix.lower()

    if suffix in {'.nc', '.tif'}:
        import xarray as xr
        import rioxarray  # noqa: F401 - registers the ``rio`` accessor

        data_coords = {
            'latitude': coords['latitude'],
            'longitude': coords['longitude'],
        }
        result_ds = xr.DataArray(
            result,
            dims=('latitude', 'longitude'),
            coords=data_coords,
            name=quantity,
        )

        result_ds.rio.set_spatial_dims(
            x_dim='longitude', y_dim='latitude', inplace=True
        )
        result_ds.rio.write_crs(4326, inplace=True)

        if suffix == '.nc':
            result_ds.to_netcdf(outfile)
        else:
            result_ds.astype('float32').rio.to_raster(outfile)
        return

    if suffix in {'.dat', '.txt'}:
        lat_grid = np.repeat(
            np.asarray(coords['latitude']).reshape(-1, 1),
            len(coords['longitude']),
            axis=1,
        )
        lon_grid = np.repeat(
            np.asarray(coords['longitude']).reshape(1, -1),
            len(coords['latitude']),
            axis=0,
        )
        values = result.reshape(-1, 1)
        out_array = np.column_stack(
            (lat_grid.ravel(), lon_grid.ravel(), values)
        )
        fmt = ['%.8f', '%.8f'] + ['%.12e'] * values.shape[1]
        np.savetxt(outfile, out_array, fmt=fmt)
        return

    raise ValueError('Not recognised output file type')


def calc_grid(params):
    print('Grid synthesis')
    _validate_params(
        params, GRID_CLI_PARAMETERS, GRID_REQUIRED_PARAMETERS
    )
    if not isinstance(params['quantity'], str):
        raise ValueError('Grid synthesis accepts exactly one quantity')
    if params['quantity'] == 'g':
        raise ValueError(
            "Grid synthesis does not support the three-component 'g' vector"
        )

    synthesis_params = {
        name: params[name]
        for name in GRID_SYNTHESIS_PARAMETERS
        if name in params
    }
    result, coords = grid_sh_synthesis(**synthesis_params)
    result = _normalize_grid_result(result, coords)
    _write_grid_output(
        params['output_file'], result, coords, params['quantity']
    )

def _normalize_point_result(result, n_points, quantity):
    result = np.asarray(result)
    if result.ndim == 1 and result.shape[0] == n_points:
        return result.reshape(-1, 1)
    if result.ndim == 2 and result.shape[0] == n_points:
        return result
    raise ValueError(
        f"Result shape for quantity {quantity!r} does not match the input points"
    )


def calc_point(params):
    print('Point synthesis')
    _validate_params(
        params, POINT_CLI_PARAMETERS, POINT_REQUIRED_PARAMETERS
    )
    input_file = params['input_file']
    output_file = params['output_file']
    point_numbers = params.get('point_numbers', True)

    data_in_file = np.atleast_2d(np.loadtxt(input_file))
    if point_numbers:
        point_coords = data_in_file[:, 1:].copy()
    else:
        point_coords = data_in_file.copy()

    if point_coords.shape[1] not in {2, 3}:
        expected = 'three or four' if point_numbers else 'two or three'
        raise ValueError(
            f'Point input must contain {expected} columns per row'
        )

    quantity_value = params['quantity']
    if isinstance(quantity_value, (list, tuple)):
        quantities = list(quantity_value)
    else:
        quantities = [quantity_value]

    if not quantities or any(not isinstance(item, str) for item in quantities):
        raise ValueError('Quantities must be specified as one or more strings')
    if len(set(quantities)) != len(quantities):
        raise ValueError('Duplicate quantities are not allowed')

    base_params = {
        name: params[name]
        for name in POINT_SYNTHESIS_PARAMETERS
        if name in params and name != 'quantity'
    }
    result_columns = []
    for quantity in quantities:
        synthesis_params = base_params.copy()
        synthesis_params['points'] = point_coords.copy()
        synthesis_params['quantity'] = quantity
        result_temp = point_sh_synthesis(**synthesis_params)
        result_columns.append(
            _normalize_point_result(
                result_temp, point_coords.shape[0], quantity
            )
        )
    result = np.hstack(result_columns)

    if point_coords.shape[1] == 2:
        height = np.zeros((data_in_file.shape[0], 1))
        data_in_file = np.hstack((data_in_file, height))

    output_array = np.hstack((data_in_file, result))
    if point_numbers:
        out_format = ['%d', '%.8f', '%.8f', '%.3f']
    else:
        out_format = ['%.8f', '%.8f', '%.3f']
    out_format.extend(['%.12e'] * result.shape[1])
    np.savetxt(output_file, output_array, fmt=out_format)



def build_parser():
    parser = argparse.ArgumentParser(description="PyHarmGrav")
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_grid = subparsers.add_parser('grid',help='Compute on grid')
    # config file
    parser_grid.add_argument('config',nargs='?', help='Path to config file')
    # options if no config file is used
    parser_grid.add_argument('--quantity',type=str)
    parser_grid.add_argument('--min_lat',type=float)
    parser_grid.add_argument('--max_lat',type=float)
    parser_grid.add_argument('--min_lon',type=float)
    parser_grid.add_argument('--max_lon',type=float)
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
    parser_grid.add_argument('--DTM_raster',type=str)
    parser_grid.add_argument('--tide_system_conversion',type=str,nargs=2)
    parser_grid.add_argument('--output_file',type=str)
    parser_grid.add_argument(
        '--normal_field_removed',
        nargs='?',
        const=True,
        type=_parse_bool,
        default=False,
    )

    parser_grid.set_defaults(
        handler=calc_grid,
        parameter_names=GRID_CLI_PARAMETERS,
        required_names=GRID_REQUIRED_PARAMETERS,
    )

    parser_point = subparsers.add_parser('point',help='Compute at scattered points')
    # config file
    parser_point.add_argument('config',nargs='?', help='Path to config file')
    # options if no config file is used
    parser_point.add_argument('--input_file',type=str)
    parser_point.add_argument('--points_type',type=str)
    parser_point.add_argument('--shcs_data',type=str)
    parser_point.add_argument('--quantity',type=str,nargs='+')
    parser_point.add_argument('--nmin',type=int,default=0)
    parser_point.add_argument('--nmax',type=int)
    parser_point.add_argument('--ellipsoid',type=str,default='GRS80')
    parser_point.add_argument('--GM',type=float)
    parser_point.add_argument('--R',type=float)
    parser_point.add_argument('--DTM_shcs_data',type=str)
    parser_point.add_argument('--DTM_raster',type=str)
    parser_point.add_argument('--point_numbers', action='store_true', default=True)
    parser_point.add_argument('--no_point_numbers', action='store_false', dest='point_numbers')
    parser_point.add_argument('--tide_system_conversion',type=str,nargs=2)
    parser_point.add_argument('--output_file',type=str)
    parser_point.add_argument(
        '--normal_field_removed',
        nargs='?',
        const=True,
        type=_parse_bool,
        default=False,
    )

    parser_point.set_defaults(
        handler=calc_point,
        parameter_names=POINT_CLI_PARAMETERS,
        required_names=POINT_REQUIRED_PARAMETERS,
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.config:
            params = load_config(args.config)
        else:
            params = {
                name: getattr(args, name)
                for name in args.parameter_names
            }
        _validate_params(
            params, args.parameter_names, args.required_names
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    args.handler(params)
    
if __name__ == '__main__':
    main()
