import numpy as np 
import pickle

from VortexAD.core.panel_method.source_doublet.doublet_functions import compute_doublet_influence
from VortexAD.core.panel_method.vortex_ring.vortex_line_functions import compute_vortex_line_ind_vel
import csdl_alpha as csdl

data_file = open('streamline_analysis_data', 'rb')
data = pickle.load(data_file)
data_file.close()

nc, ns, nt = data['grid_size'][0], data['grid_size'][1], data['grid_size'][2]
num_panels = (nc-1)*(ns-1)
mu = data['mu']
sigma = data['sigma']

mesh = data['mesh']
panel_center = data['panel_center']
local_coord_vec = data['local_coord_vec']
panel_corners = data['panel_corners']
dpij = data['dpij']
mpij = data['mij']
dij = data['dij']

wake_mesh = data['wake_mesh']
nc_w, ns_w = wake_mesh.shape[0], wake_mesh.shape[1]
num_wake_panels = (nc_w-1)*(ns_w-1)
wake_panel_center = data['wake_panel_center']
wake_local_coord_vec = data['wake_local_coord_vec']
wake_panel_corners = data['wake_panel_corners']
wake_dpij = data['wake_dpij']
wake_mpij = data['wake_mij']
wake_dij = data['wake_dij']

# x between 0.2 and 0.4
# z between 0 and 0.1
# y = 0
num_x = 15
num_z = 10
num_eval_pts = num_x*num_z
x_vec = np.linspace(0.2, 0.4, num_x)
z_vec = np.linspace(0.0, 0.1, num_z)

eval_points_mesh = np.zeros((num_x, num_z, 3))

for i in range(num_x):
    eval_points_mesh[i,:,0] = x_vec[i]
    eval_points_mesh[i,:,2] = z_vec

surface_interactions = num_panels*num_eval_pts
wake_interactions = num_wake_panels*num_eval_pts

recorder = csdl.Recorder(inline=True)
recorder.start()

# ======== INDUCED VELOCITY FROM THE SURFACE ========
eval_points_exp = csdl.expand(eval_points_mesh, (num_x, num_z, num_panels, 3), 'ijk->ijak')
eval_points_exp_vec = eval_points_exp.reshape((surface_interactions, 3))

panel_corners_exp = csdl.expand(panel_corners, (nc-1, ns-1, num_eval_pts, 3), 'ijk->ijak')
panel_corners_exp_vec = panel_corners_exp.reshape((surface_interactions, 3))

panel_x_dir = local_coord_vec[:,:,0,:]
panel_x_dir_exp = csdl.expand(panel_x_dir, (nc-1, ns-1, num_eval_pts, 3), 'ijk->ijak')
panel_x_dir_exp_vec = panel_x_dir_exp.reshape((surface_interactions, 3))

panel_y_dir = local_coord_vec[:,:,1,:]
panel_y_dir_exp = csdl.expand(panel_y_dir, (nc-1, ns-1, num_eval_pts, 3), 'ijk->ijak')
panel_y_dir_exp_vec = panel_y_dir_exp.reshape((surface_interactions, 3))

panel_normal = local_coord_vec[:,:,2,:]
panel_normal_exp = csdl.expand(panel_normal, (nc-1, ns-1, num_eval_pts, 3), 'ijk->ijak')
panel_normal_exp_vec = panel_normal_exp.reshape((surface_interactions, 3))

dpij_exp = csdl.expand(dpij, (nc-1, ns-1, num_eval_pts, 4, 3), 'ijkl->ijakl')
dpij_exp_vec = dpij_exp.reshape((surface_interactions, 4, 3))

# dij_exp = 
# dij_exp_vec = dij_exp.reshape(())

# mij_exp = 
# mij_exp_vec = mij_exp.reshape(())
