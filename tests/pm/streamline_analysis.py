import numpy as np 
import pickle
import matplotlib.pyplot as plt
import csdl_alpha as csdl

from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_influence 
from VortexAD.core.panel_method.vortex_ring.vortex_line_functions import compute_vortex_line_ind_vel


data_file = open('streamline_analysis_data', 'rb')
data = pickle.load(data_file)
data_file.close()

nc, ns, nt = 2*data['grid_size'][0]-1, data['grid_size'][1], data['grid_size'][2]
num_panels = (nc-1)*(ns-1)
mu = data['mu']
sigma = data['sigma']

mesh = data['mesh']
panel_center = data['panel_center']
local_coord_vec = data['local_coord_vec']
panel_corners = data['panel_corners']
dpij = data['dpij']
mij = data['mij']
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
num_x = 50
num_z = 50
num_eval_pts = num_x*num_z
x_vec = np.linspace(0.2, 0.4, num_x)
z_vec = np.linspace(0.05, 0.07, num_z)

eval_points_mesh = np.zeros((num_x, num_z, 3))

for i in range(num_x):
    eval_points_mesh[i,:,0] = x_vec[i]
    eval_points_mesh[i,:,2] = z_vec

surface_interactions = num_panels*num_eval_pts
wake_interactions = num_wake_panels*num_eval_pts

recorder = csdl.Recorder(inline=True)
recorder.start()

# ======== INDUCED VELOCITY FROM THE SURFACE ========
eval_points_exp = csdl.expand(eval_points_mesh, (num_x, num_z, num_panels, 4, 3), 'ijk->ijabk')
eval_points_exp_vec = eval_points_exp.reshape((surface_interactions, 4, 3))

panel_corners_exp = csdl.expand(panel_corners, (num_eval_pts, nc-1, ns-1, 4, 3), 'ijkl->aijkl')
panel_corners_exp_vec = panel_corners_exp.reshape((surface_interactions, 4, 3))

panel_x_dir = local_coord_vec[:,:,0,:]
panel_x_dir_exp = csdl.expand(panel_x_dir, (num_eval_pts, nc-1, ns-1, 4, 3), 'ijk->aijbk')
panel_x_dir_exp_vec = panel_x_dir_exp.reshape((surface_interactions, 4, 3))

panel_y_dir = local_coord_vec[:,:,1,:]
panel_y_dir_exp = csdl.expand(panel_y_dir, (num_eval_pts, nc-1, ns-1, 4, 3), 'ijk->aijbk')
panel_y_dir_exp_vec = panel_y_dir_exp.reshape((surface_interactions, 4, 3))

panel_normal = local_coord_vec[:,:,2,:]
panel_normal_exp = csdl.expand(panel_normal, (num_eval_pts, nc-1, ns-1, 4, 3), 'ijk->aijbk')
panel_normal_exp_vec = panel_normal_exp.reshape((surface_interactions, 4, 3))

dpij_exp = csdl.expand(dpij, (num_eval_pts, nc-1, ns-1, 4, 2), 'ijkl->aijkl')
dpij_exp_vec = dpij_exp.reshape((surface_interactions, 4, 2))

dij_exp = csdl.expand(dij, (num_eval_pts, nc-1, ns-1, 4), 'ijk->aijk')
dij_exp_vec = dij_exp.reshape((surface_interactions, 4))

mij_exp = csdl.expand(mij, (num_eval_pts, nc-1, ns-1, 4), 'ijk->aijk')
mij_exp_vec = mij_exp.reshape((surface_interactions, 4))

dp = eval_points_exp_vec - panel_corners_exp_vec

sum_ind = len(dp.shape) - 1
dx = csdl.sum(dp*panel_x_dir_exp_vec, axes=(sum_ind,))
dy = csdl.sum(dp*panel_y_dir_exp_vec, axes=(sum_ind,))
dz = csdl.sum(dp*panel_normal_exp_vec, axes=(sum_ind,))
rk = (dx**2 + dy**2 + dz**2 + 1.e-12)**0.5
ek = dx**2 + dz**2
hk = dx*dy

mij_list = [mij_exp_vec[:,ind] for ind in range(4)]
dij_list = [dij_exp_vec[:,ind] for ind in range(4)]
dpij_list = [[dpij_exp_vec[:,ind,0], dpij_exp_vec[:,ind,1]] for ind in range(4)]
ek_list = [ek[:,ind] for ind in range(4)]
hk_list = [hk[:,ind] for ind in range(4)]
rk_list = [rk[:,ind] for ind in range(4)]
dx_list = [dx[:,ind] for ind in range(4)]
dy_list = [dy[:,ind] for ind in range(4)]
dz_list = [dz[:,ind] for ind in range(4)]

# ==== doublet induced velocity ====
ind_vel_12 = compute_vortex_line_ind_vel(panel_corners_exp_vec[:,0,:].reshape((1,surface_interactions,3)),panel_corners_exp_vec[:,1,:].reshape((1,surface_interactions,3)),p_eval=eval_points_exp_vec[:,0,:].reshape((1,surface_interactions,3)), mode='wake')
ind_vel_23 = compute_vortex_line_ind_vel(panel_corners_exp_vec[:,1,:].reshape((1,surface_interactions,3)),panel_corners_exp_vec[:,2,:].reshape((1,surface_interactions,3)),p_eval=eval_points_exp_vec[:,1,:].reshape((1,surface_interactions,3)), mode='wake')
ind_vel_34 = compute_vortex_line_ind_vel(panel_corners_exp_vec[:,2,:].reshape((1,surface_interactions,3)),panel_corners_exp_vec[:,3,:].reshape((1,surface_interactions,3)),p_eval=eval_points_exp_vec[:,2,:].reshape((1,surface_interactions,3)), mode='wake')
ind_vel_41 = compute_vortex_line_ind_vel(panel_corners_exp_vec[:,3,:].reshape((1,surface_interactions,3)),panel_corners_exp_vec[:,0,:].reshape((1,surface_interactions,3)),p_eval=eval_points_exp_vec[:,3,:].reshape((1,surface_interactions,3)), mode='wake')

ind_vel_mu_AIC = ind_vel_12 + ind_vel_23 + ind_vel_34 + ind_vel_41
ind_vel_mu_AIC = ind_vel_mu_AIC.reshape((surface_interactions, 3)) # IN GLOBAL COORDINATES

# ==== doublet induced velocity ====
u_sigma, v_sigma, w_sigma = compute_source_influence(dij_list, mij_list, dpij_list, dx_list, dy_list, dz_list, rk_list, ek_list, hk_list, mode='velocity')

ind_vel_sigma_local = csdl.Variable(shape=u_sigma.shape + (3,), value=0.) # IN LOCAL COORDINATES
ind_vel_sigma_local = ind_vel_sigma_local.set(csdl.slice[:,0], value=u_sigma)
ind_vel_sigma_local = ind_vel_sigma_local.set(csdl.slice[:,1], value=v_sigma)
ind_vel_sigma_local = ind_vel_sigma_local.set(csdl.slice[:,2], value=w_sigma)

local_coord_vec_exp = csdl.expand(local_coord_vec, (nc-1, ns-1, num_eval_pts, 3, 3), 'ijkl->ijakl')
local_coord_vec_exp_vec = local_coord_vec_exp.reshape((surface_interactions, 3, 3))

local_coord_vec_exp_vec_T = csdl.Variable(shape=local_coord_vec_exp_vec.shape, value=0)

local_coord_vec_exp_vec_T = local_coord_vec_exp_vec_T.set(csdl.slice[:,0,:], value=local_coord_vec_exp_vec[:,:,0])
local_coord_vec_exp_vec_T = local_coord_vec_exp_vec_T.set(csdl.slice[:,:,0], value=local_coord_vec_exp_vec[:,0,:])
local_coord_vec_exp_vec_T = local_coord_vec_exp_vec_T.set(csdl.slice[:,2,:], value=local_coord_vec_exp_vec[:,:,2])
local_coord_vec_exp_vec_T = local_coord_vec_exp_vec_T.set(csdl.slice[:,:,2], value=local_coord_vec_exp_vec[:,2,:])
local_coord_vec_exp_vec_T = local_coord_vec_exp_vec_T.set(csdl.slice[:,1,1], value=local_coord_vec_exp_vec[:,1,1])

ind_vel_sigma_AIC = csdl.einsum(ind_vel_sigma_local, local_coord_vec_exp_vec_T, action='ij,ikj->ik')

# mat-vec to get total induced velocity
ind_vel_mu_AIC_grid = ind_vel_mu_AIC.reshape((num_eval_pts, num_panels, 3))
ind_vel_sigma_AIC_grid = ind_vel_sigma_AIC.reshape((num_eval_pts, num_panels, 3))

ind_vel_mu = csdl.Variable(shape=(num_eval_pts, 3), value=0.)
ind_vel_sigma = csdl.Variable(shape=(num_eval_pts, 3), value=0.)

for i in range(3):
    ind_vel_mu = ind_vel_mu.set(csdl.slice[:,i], value=csdl.matvec(ind_vel_mu_AIC_grid[:,:,i], mu.reshape(num_panels,)))
    ind_vel_sigma = ind_vel_sigma.set(csdl.slice[:,i], value=csdl.matvec(ind_vel_sigma_AIC_grid[:,:,i], sigma.reshape(num_panels,)))

surf_induced_vel = ind_vel_mu + ind_vel_sigma
# surf_induced_vel = ind_vel_mu

surf_induced_vel_grid = surf_induced_vel.reshape((num_x, num_z, 3))

asdf = np.zeros(shape=surf_induced_vel_grid.shape)
asdf[:,:,0] = 10.


ind_vel_mag = csdl.norm(surf_induced_vel_grid + asdf, axes=(2,))


X, Z = np.meshgrid(x_vec, z_vec)

panel_center_x = [1.42857143e-01, 2.14285714e-01, 2.85714286e-01, 3.57142857e-01, 4.28571429e-01]
panel_center_z = [5.26912935e-02, 5.81154169e-02, 5.99456941e-02, 5.93067998e-02, 5.68255165e-02]


fig, ax = plt.subplots()
# CS = ax.contour(X, Z, ind_vel_mag.value.T)
strm = ax.streamplot(X, Z, surf_induced_vel_grid.value[:,:,0] + 10, surf_induced_vel_grid.value[:,:,2], color=ind_vel_mag.value,)
fig.colorbar(strm.lines)
ax.plot(panel_center_x, panel_center_z, 'k-o', linewidth=3)
ax.set_title('Simplest default with labels')
ax.set_xlim([x_vec[0], x_vec[-1]])
ax.set_ylim([z_vec[0], z_vec[-1]])


plt.show()

