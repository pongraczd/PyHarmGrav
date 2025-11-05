from pyharm_grav import point_sh_synthesis, grid_sh_synthesis
import numpy as np

gpm_path = "./sample_input_data/EGM96.mat"
points_path = "./sample_input_data/sample_points.txt"
dtm_path = "./sample_input_data/DTM2006.mat"
points0_path = "./sample_input_data/sample_points_h0.txt"
GM = 3986004.415E+8
R = 6378136.3

points = np.loadtxt(points_path)[:,1:]
points0 = np.loadtxt(points0_path)

# example 1: geoid undulation synthesis on grid , nmax not specified, model up to nmax=2190
print('geoid')
result = grid_sh_synthesis('N',45,49,15,20,1,gpm_path,'minutes',0,None,'GRS80',None,'ellipsoid',0,None,GM,R,dtm_path,None)
print(result)
# example 2: topography synthesis on grid , nmax = 6500 , model up to nmax=10800
topo = grid_sh_synthesis('topo',45,49,15,20,1,dtm_path,'minutes',0,300,'GRS80',None,'ellipsoid',0,None,1,1,None,None)
print(topo) # passed without error
# example 3
result2 = point_sh_synthesis(points0,gpm_path,'ellipsoidal','N',None,0,None,'GRS80',GM,R,dtm_path)
print(result2)
#example 4: point synthesis with nmin + nmax, dg
result3 = point_sh_synthesis(points,gpm_path,'ellipsoidal','dg',None,10,None,'GRS80',GM,R,dtm_path)
print(result3)
#TODO :  , log file, create GUI