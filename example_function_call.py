import numpy as np

from pyharmgrav.gshs import grid_sh_synthesis, point_sh_synthesis


gpm_path = "./input_data/EGM96.mat"
points_path = "./input_data/sample_points.txt"
dtm_path = "./input_data/DTM2006.mat"
points0_path = "./input_data/sample_points_h0.txt"
GM = 3986004.415e8
R = 6378136.3

points = np.loadtxt(points_path)[:, 1:]
points0 = np.loadtxt(points0_path)

# Example 1: geoid undulation on a grid; nmax is inferred as 360.
print("geoid")
geoid, geoid_coords = grid_sh_synthesis(
    quantity="N",
    min_lat=45,
    max_lat=49,
    min_lon=15,
    max_lon=20,
    resolution=1,
    shcs_data=gpm_path,
    resolution_unit="minutes",
    nmin=0,
    nmax=None,
    ellipsoid="GRS80",
    ref_surface_type="ellipsoid",
    height=0,
    GM=GM,
    R=R,
    DTM_shcs_data=dtm_path,
    normal_field_removed=False,
)
print(geoid)

# Example 2: topography on a grid, truncated to degree 300 (model nmax=360).
topo, topo_coords = grid_sh_synthesis(
    quantity="topo",
    min_lat=45,
    max_lat=49,
    min_lon=15,
    max_lon=20,
    resolution=1,
    shcs_data=dtm_path,
    resolution_unit="minutes",
    nmin=0,
    nmax=300,
    ellipsoid="GRS80",
    ref_surface_type="ellipsoid",
    height=0,
    GM=1,
    R=1,
    normal_field_removed=False,
)
print(topo)

# Example 3: geoid undulation at zero-height ellipsoidal points.
result2 = point_sh_synthesis(
    points=points0,
    shcs_data=gpm_path,
    points_type="ellipsoidal",
    quantity="N",
    nmin=0,
    nmax=None,
    ellipsoid="GRS80",
    GM=GM,
    R=R,
    DTM_shcs_data=dtm_path,
    normal_field_removed=False,
)
print(result2)

# Example 4: gravity anomaly from degree 10 through the model's maximum degree.
result3 = point_sh_synthesis(
    points=points,
    shcs_data=gpm_path,
    points_type="ellipsoidal",
    quantity="dg",
    nmin=10,
    nmax=None,
    ellipsoid="GRS80",
    GM=GM,
    R=R,
    normal_field_removed=False,
)
print(result3)

# Example 5: north component of the vertical deflection.
result4 = point_sh_synthesis(
    points=points,
    shcs_data=gpm_path,
    points_type="ellipsoidal",
    quantity="xi",
    nmin=0,
    nmax=None,
    ellipsoid="GRS80",
    GM=GM,
    R=R,
    normal_field_removed=False,
)
print(result4)

# Example 6: generalized height anomaly at nonzero ellipsoidal heights.
result5 = point_sh_synthesis(
    points=points,
    shcs_data=gpm_path,
    points_type="ellipsoidal",
    quantity="zeta_ell",
    nmin=0,
    nmax=None,
    ellipsoid="GRS80",
    GM=GM,
    R=R,
    normal_field_removed=False,
)
print(result5)
