import numpy as np
import csdl_alpha as csdl

import csdl_alpha as csdl
import numpy as np 
from VortexAD.core.geometry.gen_panel_mesh import gen_panel_mesh, gen_panel_mesh_new
from VortexAD import steady_panel_solver

# plotting functions
import matplotlib.pyplot as plt
from VortexAD.utils.plot import plot_pressure_distribution

from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_influence_new
from VortexAD.core.panel_method.vortex_ring.vortex_line_functions import compute_vortex_line_ind_vel

# off body analysis function
def off_body_analysis(mesh_dict, wake_mesh_dict, eval_points, mu, sigma, velocity_grid):
    eval_pts_shape = eval_points.shape
    nc_e, ns_e = eval_pts_shape[1], eval_pts_shape[2]
    num_eval_pts = nc_e*ns_e
    eval_pts = eval_points[0,:]

    surface_names = list(mesh_dict.keys())

    start, stop = 0, 0
    start_w, stop_w = 0, 0
    num_tot_panels, num_tot_wake_panels = 0, 0
    for i in range(len(surface_names)):
        surf_i_name = surface_names[i]
        num_panels = mesh_dict[surf_i_name]['num_panels']
        num_wake_panels = wake_mesh_dict[surf_i_name]['num_panels']

        num_tot_panels += num_panels
        num_tot_wake_panels += num_wake_panels


    start_i, stop_i = 0, 0
    start_i_w, stop_i_w = 0, 0

    AIC_mu = csdl.Variable(value=np.zeros((num_eval_pts, num_tot_panels, 3)))
    AIC_mu_wake = csdl.Variable(value=np.zeros((num_eval_pts, num_tot_wake_panels, 3)))
    AIC_sigma = csdl.Variable(value=np.zeros((num_eval_pts, num_tot_panels, 3)))

    mu_wake = csdl.Variable(value=np.zeros((num_tot_wake_panels,)))
    
    for i in range(len(surface_names)):
        surf_i_name = surface_names[i]

        # ==================== assembling AIC for surface doublets and sources ====================
        nc_i, ns_i = mesh_dict[surf_i_name]['nc'], mesh_dict[surf_i_name]['ns']
        num_panels_i = mesh_dict[surf_i_name]['num_panels']
        stop_i += num_panels_i

        mu_surf_grid = mu[start_i:stop_i].reshape((nc_i-1, ns_i-1))
        mu_surf_wake = mu_surf_grid[-1,:] - mu_surf_grid[0,:]

        panel_corners_i = mesh_dict[surf_i_name]['panel_corners'][0,:]
        coll_point_i = mesh_dict[surf_i_name]['panel_center'][0,:]
        panel_x_dir_i = mesh_dict[surf_i_name]['panel_x_dir'][0,:]
        panel_y_dir_i = mesh_dict[surf_i_name]['panel_y_dir'][0,:]
        panel_normal_i = mesh_dict[surf_i_name]['panel_normal'][0,:]

        S_i = mesh_dict[surf_i_name]['S'][0,:]
        SL_i = mesh_dict[surf_i_name]['SL'][0,:]
        SM_i = mesh_dict[surf_i_name]['SM'][0,:]
        
        num_surf_interactions = num_eval_pts*num_panels_i

        eval_pt_exp = csdl.expand(eval_pts, (nc_e, ns_e, num_panels_i, 4, 3), 'ijk->ijabk')
        eval_pt_exp_vec = eval_pt_exp.reshape((num_surf_interactions, 4, 3))

        panel_corners_i_exp = csdl.expand(panel_corners_i, (num_eval_pts, nc_i-1, ns_i-1, 4, 3), 'ijkl->aijkl')
        panel_corners_i_exp_vec = panel_corners_i_exp.reshape((num_surf_interactions, 4, 3))

        coll_point_i_exp = csdl.expand(coll_point_i, (num_eval_pts, nc_i-1, ns_i-1, 4, 3), 'ijk->aijbk')
        coll_point_i_exp_vec = coll_point_i_exp.reshape((num_surf_interactions, 4, 3))

        panel_x_dir_i_exp = csdl.expand(panel_x_dir_i, (num_eval_pts, nc_i-1, ns_i-1, 4, 3), 'ijk->aijbk')
        panel_x_dir_i_exp_vec = panel_x_dir_i_exp.reshape((num_surf_interactions, 4, 3))

        panel_y_dir_i_exp = csdl.expand(panel_y_dir_i, (num_eval_pts, nc_i-1, ns_i-1, 4, 3), 'ijk->aijbk')
        panel_y_dir_i_exp_vec = panel_y_dir_i_exp.reshape((num_surf_interactions, 4, 3))

        panel_normal_i_exp = csdl.expand(panel_normal_i, (num_eval_pts, nc_i-1, ns_i-1, 4, 3), 'ijk->aijbk')
        panel_normal_i_exp_vec = panel_normal_i_exp.reshape((num_surf_interactions, 4, 3))

        S_i_exp = csdl.expand(S_i, (num_eval_pts, nc_i-1, ns_i-1, 4), 'ijk->aijk')
        S_i_exp_vec = S_i_exp.reshape((num_surf_interactions, 4))

        SL_i_exp = csdl.expand(SL_i, (num_eval_pts, nc_i-1, ns_i-1, 4), 'ijk->aijk')
        SL_i_exp_vec = SL_i_exp.reshape((num_surf_interactions, 4))

        SM_i_exp = csdl.expand(SM_i, (num_eval_pts, nc_i-1, ns_i-1, 4), 'ijk->aijk')
        SM_i_exp_vec = SM_i_exp.reshape((num_surf_interactions, 4))

        a = eval_pt_exp_vec - panel_corners_i_exp_vec # Rc - Ri
        P_JK = eval_pt_exp_vec - coll_point_i_exp_vec # RcJ - RcK
        sum_ind = len(a.shape) - 1

        A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
        AL = csdl.sum(a*panel_x_dir_i_exp_vec, axes=(sum_ind,))
        AM = csdl.sum(a*panel_y_dir_i_exp_vec, axes=(sum_ind,)) # m-direction projection 
        PN = csdl.sum(P_JK*panel_normal_i_exp_vec, axes=(sum_ind,)) # normal projection of CP

        B = csdl.Variable(shape=A.shape, value=0.)
        B = B.set(csdl.slice[:,:-1], value=A[:,1:])
        B = B.set(csdl.slice[:,-1], value=A[:,0])

        BL = csdl.Variable(shape=AL.shape, value=0.)
        BL = BL.set(csdl.slice[:,:-1], value=BL[:,1:])
        BL = BL.set(csdl.slice[:,-1], value=BL[:,0])

        BM = csdl.Variable(shape=AM.shape, value=0.)
        BM = BM.set(csdl.slice[:,:-1], value=AM[:,1:])
        BM = BM.set(csdl.slice[:,-1], value=AM[:,0])

        A1 = AM*SL_i_exp_vec - AL*SM_i_exp_vec

        # print(A.shape)
        A = A.expand(panel_normal_i_exp_vec.shape, 'ij->ija')
        AM = AM.expand(panel_normal_i_exp_vec.shape, 'ij->ija')
        B = B.expand(panel_normal_i_exp_vec.shape, 'ij->ija')
        BM = BM.expand(panel_normal_i_exp_vec.shape, 'ij->ija')
        SL_i_exp_vec = SL_i_exp_vec.expand(panel_normal_i_exp_vec.shape, 'ij->ija')
        SM_i_exp_vec = SM_i_exp_vec.expand(panel_normal_i_exp_vec.shape, 'ij->ija')
        A1 = A1.expand(panel_normal_i_exp_vec.shape, 'ij->ija')
        PN = PN.expand(panel_normal_i_exp_vec.shape, 'ij->ija')
        S_i_exp_vec = S_i_exp_vec.expand(panel_normal_i_exp_vec.shape, 'ij->ija')

        A_list = [A[:,ind] for ind in range(4)]
        AM_list = [AM[:,ind] for ind in range(4)]
        B_list = [B[:,ind] for ind in range(4)]
        BM_list = [BM[:,ind] for ind in range(4)]
        SL_list = [SL_i_exp_vec[:,ind] for ind in range(4)]
        SM_list = [SM_i_exp_vec[:,ind] for ind in range(4)]
        A1_list = [A1[:,ind] for ind in range(4)]
        PN_list = [PN[:,ind] for ind in range(4)]
        S_list = [S_i_exp_vec[:,ind] for ind in range(4)]

        ind_vel_source_local = compute_source_influence_new(
            A_list, 
            AM_list, 
            B_list, 
            BM_list, 
            SL_list, 
            SM_list, 
            A1_list, 
            PN_list, 
            S_list, 
            panel_x_dir_i_exp_vec[:,0,:], # these don't change along dimension 2
            panel_y_dir_i_exp_vec[:,0,:],
            panel_normal_i_exp_vec[:,0,:],
            mode='velocity'
        )

        rot_mat_s_j = mesh_dict[surf_i_name]['rot_mat'][0,:,:,:,:]
        rot_mat_s_j_exp = csdl.expand(rot_mat_s_j, (num_eval_pts, nc_i-1, ns_i-1, 3, 3), 'ijkl->aijkl')
        rot_mat_s_j_exp_vec = rot_mat_s_j_exp.reshape(((num_surf_interactions, 3, 3)))

        # ind_vel_source_global = csdl.einsum(ind_vel_source_local, rot_mat_s_j_exp_vec, action='ijk,ijlk->ijl')
        ind_vel_source_global = ind_vel_source_local
        ind_vel_source_global_mat = ind_vel_source_global.reshape((num_eval_pts, num_panels_i, 3))

        AIC_sigma = AIC_sigma.set(csdl.slice[:,start_i:stop_i,:], value=ind_vel_source_global_mat)

        panel_corners_vr = panel_corners_i_exp_vec.reshape((1,) + panel_corners_i_exp_vec.shape)
        eval_pt_vr = eval_pt_exp_vec.reshape((1,) + eval_pt_exp_vec.shape)

        ind_vel_s_12 = compute_vortex_line_ind_vel(panel_corners_vr[:,:,0,:],panel_corners_vr[:,:,1,:],p_eval=eval_pt_vr[:,:,0,:], mode='wake', vc=1.e-3)
        ind_vel_s_23 = compute_vortex_line_ind_vel(panel_corners_vr[:,:,1,:],panel_corners_vr[:,:,2,:],p_eval=eval_pt_vr[:,:,0,:], mode='wake', vc=1.e-3)
        ind_vel_s_34 = compute_vortex_line_ind_vel(panel_corners_vr[:,:,2,:],panel_corners_vr[:,:,3,:],p_eval=eval_pt_vr[:,:,0,:], mode='wake', vc=1.e-3)
        ind_vel_s_41 = compute_vortex_line_ind_vel(panel_corners_vr[:,:,3,:],panel_corners_vr[:,:,0,:],p_eval=eval_pt_vr[:,:,0,:], mode='wake', vc=1.e-3)

        ind_vel_s = ind_vel_s_12+ind_vel_s_23+ind_vel_s_34+ind_vel_s_41 # (nn, num_interactions, 3)
        ind_vel_s = ind_vel_s[0,:] # removing num_nodes
        ind_vel_s_mat = ind_vel_s.reshape((num_eval_pts, num_panels_i, 3))
        AIC_mu = AIC_mu.set(csdl.slice[:,start_i:stop_i,:], value=ind_vel_s_mat)

        # ==================== assembling AIC for wake doublets ====================
        nc_i_w, ns_i_w = wake_mesh_dict[surf_i_name]['nc'], wake_mesh_dict[surf_i_name]['ns']
        num_panels_i_w = wake_mesh_dict[surf_i_name]['num_panels']
        stop_i_w += num_panels_i_w

        panel_corners_i = wake_mesh_dict[surf_i_name]['panel_corners'][0,:]
        
        num_wake_interactions = num_eval_pts*num_panels_i_w

        eval_pt_w_exp = csdl.expand(eval_pts, (nc_e, ns_e, num_panels_i_w, 4, 3), 'ijk->ijabk')
        eval_pt_w_exp_vec = eval_pt_w_exp.reshape((num_wake_interactions, 4, 3))

        panel_corners_i_w_exp = csdl.expand(panel_corners_i, (num_eval_pts, nc_i_w-1, ns_i_w-1, 4, 3), 'ijkl->aijkl')
        panel_corners_i_w_exp_vec = panel_corners_i_w_exp.reshape((num_wake_interactions, 4, 3))

        panel_corners_vr = panel_corners_i_w_exp_vec.reshape((1,) + panel_corners_i_w_exp_vec.shape)
        eval_pt_vr = eval_pt_w_exp_vec.reshape((1,) + eval_pt_w_exp_vec.shape)

        ind_vel_s_12 = compute_vortex_line_ind_vel(panel_corners_vr[:,:,0,:],panel_corners_vr[:,:,1,:],p_eval=eval_pt_vr[:,:,0,:], mode='wake', vc=1.e-3)
        ind_vel_s_23 = compute_vortex_line_ind_vel(panel_corners_vr[:,:,1,:],panel_corners_vr[:,:,2,:],p_eval=eval_pt_vr[:,:,0,:], mode='wake', vc=1.e-3)
        ind_vel_s_34 = compute_vortex_line_ind_vel(panel_corners_vr[:,:,2,:],panel_corners_vr[:,:,3,:],p_eval=eval_pt_vr[:,:,0,:], mode='wake', vc=1.e-3)
        ind_vel_s_41 = compute_vortex_line_ind_vel(panel_corners_vr[:,:,3,:],panel_corners_vr[:,:,0,:],p_eval=eval_pt_vr[:,:,0,:], mode='wake', vc=1.e-3)

        ind_vel_s = ind_vel_s_12+ind_vel_s_23+ind_vel_s_34+ind_vel_s_41 # (nn, num_interactions, 3)
        ind_vel_s = ind_vel_s[0,:] # removing num_nodes
        ind_vel_s_mat = ind_vel_s.reshape((num_eval_pts, num_panels_i_w, 3))
        AIC_mu = AIC_mu.set(csdl.slice[:,start_i_w:stop_i_w,:], value=ind_vel_s_mat)

        mu_wake = mu_wake.set(csdl.slice[start_i_w:stop_i_w], value=mu_surf_wake.reshape((num_panels_i_w,)))

        start_i += num_panels_i
        start_i_w += num_panels_i_w

    induced_vel = csdl.Variable(shape=(num_eval_pts, 3), value=0.)

    for direction in csdl.frange(3):
        sigma_induced_vel = csdl.matvec(AIC_sigma[:,:,direction], sigma)

        mu_surf_induced_vel = csdl.matvec(AIC_mu[:,:,direction], mu)
        mu_wake_induced_vel = csdl.matvec(AIC_mu_wake[:,:,direction], mu_wake)

        total_induced_vel = sigma_induced_vel + mu_surf_induced_vel + mu_wake_induced_vel
        induced_vel = induced_vel.set(csdl.slice[:,direction], value=total_induced_vel)

    induced_vel_grid = induced_vel.reshape((nc_e, ns_e, 3))
    total_vel = induced_vel_grid + velocity_grid

    Q_inf_norm = csdl.norm(velocity_grid, axes=(2,))
    Q_pert_norm = csdl.norm(total_vel, axes=(2,))

    Cp = 1 - Q_pert_norm**2/Q_inf_norm**2

    Cp = Cp.reshape((1,) + Cp.shape)

    return Cp, Q_pert_norm, Q_inf_norm

# setting up inputs
b = 10.
c = 1.
num_nodes = 1
alpha_deg = 0.
alpha = np.deg2rad(alpha_deg) # aoa
mach = 0.15
sos = 340.3
Vx = sos*mach
V_inf = np.array([-Vx, 0., 0.])

# coarse and fine grids
ns_fine, nc_fine = 21, 31
ns_coarse, nc_coarse = 11, 21

mesh_orig_fine = gen_panel_mesh(nc_fine, ns_fine, c, b, span_spacing='cosine',  frame='default', plot_mesh=False) # even chordwise spacing
mesh_orig_coarse = gen_panel_mesh(nc_coarse, ns_coarse, c, b, span_spacing='cosine',  frame='default', plot_mesh=False) # even chordwise spacing

mesh_orig_fine = gen_panel_mesh_new(nc_fine, ns_fine, c, b,  frame='default', plot_mesh=False) # uneven chordwise spacing
mesh_orig_coarse = gen_panel_mesh_new(nc_coarse, ns_coarse, c, b,  frame='default', plot_mesh=False) # uneven chordwise spacing

mesh_fine = np.zeros((num_nodes,) + mesh_orig_fine.shape)
mesh_coarse = np.zeros((num_nodes,) + mesh_orig_coarse.shape)
for i in range(num_nodes):
    mesh_fine[i,:] = mesh_orig_fine
    mesh_coarse[i,:] = mesh_orig_coarse

V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

mesh_velocity_fine = np.zeros_like(mesh_fine)
mesh_vel_coarse = np.zeros_like(mesh_coarse)
for i in range(num_nodes):
    mesh_velocity_fine[i,:] = V_inf_rot
    mesh_vel_coarse[i,:] = V_inf_rot

# want coarse data at panel centers
coll_pt_coarse = (mesh_coarse[:,:-1,:-1] + mesh_coarse[:,1:,:-1] + mesh_coarse[:,1:,1:] + mesh_coarse[:,:-1,1:])/4
coll_vel_coarse = (mesh_vel_coarse[:,:-1,:-1] + mesh_vel_coarse[:,1:,:-1] + mesh_vel_coarse[:,1:,1:] + mesh_vel_coarse[:,:-1,1:])/4

recorder = csdl.Recorder(inline=False)
recorder.start()

# fine grid variables
mesh_fine = csdl.Variable(value=mesh_fine)
mesh_velocity_fine = csdl.Variable(value=mesh_velocity_fine)
# coarse grid variables
coll_pt_coarse = csdl.Variable(value=coll_pt_coarse)
coll_vel_coarse = csdl.Variable(value=coll_vel_coarse)

mesh_list = [mesh_fine]
mesh_velocity_list = [mesh_velocity_fine]

# running original solver with fine grid
output_dict, mesh_dict, mu, sigma = steady_panel_solver(
    mesh_list, 
    mesh_velocity_list
)

wake_mesh_dict = output_dict['wake_dict']

Cp_fine = output_dict['surface_0']['Cp']
CL_fine  = output_dict['surface_0']['CL']
CDi_fine = output_dict['surface_0']['CDi']

# doing off-body analysis on coarse grid via superposition
Cp_coarse, Q, Q_inf = off_body_analysis(mesh_dict, wake_mesh_dict, coll_vel_coarse, mu[0,:], sigma[0,:], coll_vel_coarse[0,:])

recorder.stop()
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[mesh_fine, mesh_velocity_fine], # list of outputs (put in csdl variable)
    additional_outputs=[Cp_fine, Cp_coarse, CL_fine, CDi_fine], # list of outputs (put in csdl variable)
)
jax_sim.run()

mesh_fine = jax_sim[mesh_fine]
Cp_fine = jax_sim[Cp_fine]
Cp_coarse = jax_sim[Cp_coarse]
CL_fine = jax_sim[CL_fine]
CDi_fine = jax_sim[CDi_fine]


print(f'CL: {CL_fine}')
print(f'CDi: {CDi_fine}')


if True:
    plot_pressure_distribution([mesh_fine], [Cp_fine], interactive=True, top_view=False)

if True:
    plot_pressure_distribution([mesh_coarse], [Cp_coarse], interactive=True, top_view=False)