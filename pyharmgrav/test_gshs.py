import time
from pathlib import Path
import numpy as np
import pyharm as ph
from pyharmgrav.gshs_shbundle import gshs_point

nmax= 200
max_workers = 8
def main():
    model_path = Path(__file__).resolve().parents[1] / "input_data" / "EIGEN-6C4.gfc"
    shcs = ph.shc.Shc.from_file("gfc", str(model_path), nmax)

    row = 100
    col = 200
    lon_vec = np.linspace(-np.pi, np.pi, col)
    lat_vec = np.linspace(-np.pi / 2, np.pi / 2, row)
    lon, lat = np.meshgrid(lon_vec,lat_vec)

    lon = lon.ravel()
    lat = lat.ravel()
    radius = np.full(lat.shape, 7e6)
    

    start = time.perf_counter()
    single_result = gshs_point(None,shcs=shcs,lat=lat,lon=lon,r=radius,nmax=None,error=False,type="sctr",n_workers=1)
    single_time = time.perf_counter() - start

    start = time.perf_counter()
    parallel_result = gshs_point(None,shcs=shcs,lat=lat, lon=lon, r=radius,nmax=None, error=False,type="sctr",n_workers=max_workers)
    parallel_time = time.perf_counter() - start

    grid_time_start = time.perf_counter()
    parallel_result_grid = gshs_point(None,shcs=shcs,lat=lat_vec, lon=lon_vec, r=7e6,nmax=None, error=False,type="grid",n_workers=max_workers)
    grid_time = time.perf_counter() - grid_time_start
    np.testing.assert_allclose(parallel_result, single_result, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(parallel_result_grid, parallel_result.reshape((row,col)), rtol=1e-12, atol=1e-8)
    speedup = single_time / parallel_time
    efficiency = speedup / max_workers

    print(f"Single-worker time: {single_time:.4f} seconds")
    print(f"{max_workers}-worker time: {parallel_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Parallel efficiency: {efficiency:.1%}")
    print(f"Grid time: {grid_time:.4f} seconds")

    pnt = ph.crd.PointSctr.from_arrays(lat=lat, lon=lon, r=radius)
    start = time.perf_counter()
    res2 = ph.shs.point(pnt, shcs=shcs,nmax=nmax)
    ph_time = time.perf_counter() - start
    print(f"PyHarm time: {ph_time:.4f} seconds")
    print(f"Max difference between PyHarm and parallel result: {np.max(np.abs(res2-parallel_result))}")
    print(res2-parallel_result)

if __name__ == "__main__":
    main()
