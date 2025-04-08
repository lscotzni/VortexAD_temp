import numpy as np
import csdl_alpha as csdl
import pickle
from VortexAD.core.geometry.gen_panel_mesh import gen_panel_mesh, gen_panel_mesh_new
from VortexAD import steady_panel_solver

# plotting functions
import matplotlib.pyplot as plt
from VortexAD.utils.plot import plot_pressure_distribution

from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_influence_new
from VortexAD.core.panel_method.vortex_ring.vortex_line_functions import compute_vortex_line_ind_vel

# off body analysis function
def off_body_analysis(mesh_dict, wake_mesh_dict, eval_points, mu, sigma, velocity):
    eval_pts_shape = eval_points.shape # num_points, 3
    num_eval_pts = eval_pts_shape[0]
    eval_pts = eval_points[:,:]

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
        print(panel_corners_i.shape)
        coll_point_i = mesh_dict[surf_i_name]['panel_center'][0,:]
        panel_x_dir_i = mesh_dict[surf_i_name]['panel_x_dir'][0,:]
        panel_y_dir_i = mesh_dict[surf_i_name]['panel_y_dir'][0,:]
        panel_normal_i = mesh_dict[surf_i_name]['panel_normal'][0,:]

        S_i = mesh_dict[surf_i_name]['S'][0,:]
        SL_i = mesh_dict[surf_i_name]['SL'][0,:]
        SM_i = mesh_dict[surf_i_name]['SM'][0,:]
        
        num_surf_interactions = num_eval_pts*num_panels_i

        eval_pt_exp = csdl.expand(eval_pts, (num_eval_pts, num_panels_i, 4, 3), 'ij->iabj')
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

        # rot_mat_s_j = mesh_dict[surf_i_name]['rot_mat'][0,:,:,:]
        # rot_mat_s_j_exp = csdl.expand(rot_mat_s_j, (num_eval_pts, nc_i-1, ns_i-1, 3, 3), 'ijkl->aijkl')
        # rot_mat_s_j_exp_vec = rot_mat_s_j_exp.reshape(((num_surf_interactions, 3, 3)))

        # ind_vel_source_global = csdl.einsum(ind_vel_source_local, rot_mat_s_j_exp_vec, action='ij,ilj->il')
        ind_vel_source_global = ind_vel_source_local
        ind_vel_source_global_mat = ind_vel_source_global.reshape((num_eval_pts, num_panels_i, 3))

        AIC_sigma = AIC_sigma.set(csdl.slice[:,start_i:stop_i,:], value=ind_vel_source_global_mat)

        panel_corners_vr = panel_corners_i_exp_vec.reshape((1,) + panel_corners_i_exp_vec.shape)
        eval_pt_vr = eval_pt_exp_vec.reshape((1,) + eval_pt_exp_vec.shape)

        ind_vel_s_12 = compute_vortex_line_ind_vel(panel_corners_vr[:,:,0,:],panel_corners_vr[:,:,1,:],p_eval=eval_pt_vr[:,:,0,:], mode='wake', vc=1.e-4)
        ind_vel_s_23 = compute_vortex_line_ind_vel(panel_corners_vr[:,:,1,:],panel_corners_vr[:,:,2,:],p_eval=eval_pt_vr[:,:,0,:], mode='wake', vc=1.e-4)
        ind_vel_s_34 = compute_vortex_line_ind_vel(panel_corners_vr[:,:,2,:],panel_corners_vr[:,:,3,:],p_eval=eval_pt_vr[:,:,0,:], mode='wake', vc=1.e-4)
        ind_vel_s_41 = compute_vortex_line_ind_vel(panel_corners_vr[:,:,3,:],panel_corners_vr[:,:,0,:],p_eval=eval_pt_vr[:,:,0,:], mode='wake', vc=1.e-4)

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

        eval_pt_w_exp = csdl.expand(eval_pts, (num_eval_pts, num_panels_i_w, 4, 3), 'ij->iabj')
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
        AIC_mu_wake = AIC_mu_wake.set(csdl.slice[:,start_i_w:stop_i_w,:], value=ind_vel_s_mat)

        mu_grid = mu[start_i:stop_i].reshape((nc_i-1, ns_i-1))
        mu_surf_wake = mu_grid[-1,:] - mu_grid[0,:]

        mu_wake = mu_wake.set(csdl.slice[start_i_w:stop_i_w], value=mu_surf_wake.reshape((num_panels_i_w,)))

        start_i += num_panels_i
        start_i_w += num_panels_i_w

    induced_vel = csdl.Variable(shape=(num_eval_pts, 3), value=0.)

    for direction in csdl.frange(3):
        sigma_induced_vel = csdl.matvec(AIC_sigma[:,:,direction], sigma) # asymmetric @ 0 aoa :/

        mu_surf_induced_vel = csdl.matvec(AIC_mu[:,:,direction], mu) # symmetric as should be at 0 aoa
        mu_wake_induced_vel = csdl.matvec(AIC_mu_wake[:,:,direction], mu_wake) # this is causing some asymmetry at the TE

        total_induced_vel = sigma_induced_vel + mu_surf_induced_vel + mu_wake_induced_vel
        # total_induced_vel = mu_surf_induced_vel + mu_wake_induced_vel
        induced_vel = induced_vel.set(csdl.slice[:,direction], value=total_induced_vel)

    velocity_vec = velocity.expand(induced_vel.shape, 'i->ji')
    total_vel = induced_vel + velocity_vec

    Q_inf_norm = csdl.norm(velocity_vec, axes=(1,))
    Q_pert_norm = csdl.norm(total_vel, axes=(1,))

    Cp = 1 - Q_pert_norm**2/Q_inf_norm**2

    Cp = Cp.reshape((1,) + Cp.shape)

    return Cp, total_vel

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

with open('superposition_data.pickle', 'rb') as file:
    data = pickle.load(file)

recorder = csdl.Recorder(inline=False)
recorder.start()

mesh_dict = data['mesh_dict']
nc, ns = mesh_dict['surface_0']['nc'], mesh_dict['surface_0']['ns']
wake_mesh_dict = data['wake_mesh_dict']
mu = data['mu']
sigma = data['sigma']
Cp = data['Cp']
mesh = data['mesh']
mesh_velocity = csdl.Variable(value=data['mesh_velocity'])
free_stream_vel = mesh_velocity[0,0,0,:] * -1. # reference frame sign change

num_eval_pts = 1
eval_pts = csdl.Variable(value=1., shape=(num_eval_pts, 3))
eval_pts = csdl.Variable(value=np.array([0.49774775, 0.        , -0.05305305]).reshape((1,3)))

eval_pt_Cp, eval_pt_velocity = off_body_analysis(mesh_dict, wake_mesh_dict, eval_pts, mu[0,:], sigma[0,:], free_stream_vel)

inputs = [
    eval_pts,
]

outputs = [
    eval_pt_Cp,
    eval_pt_velocity,
    free_stream_vel
]

recorder.stop()
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=eval_pts, # list of outputs (put in csdl variable)
    additional_outputs=outputs, # list of outputs (put in csdl variable)
)

# jax_sim.run()
# 
# free_stream_vel = jax_sim[free_stream_vel]

if False:
    sample_eval_pt = np.array([0.49774775, 0.        , -0.05305305]).reshape((1,3))
    jax_sim[eval_pts] = sample_eval_pt
    jax_sim.run()
    sample_velocity = jax_sim[eval_pt_velocity]
    exit()

if False:
    plot_pressure_distribution([mesh], [Cp], interactive=True, top_view=False)

jax_sim.run()

free_stream_vel = jax_sim[free_stream_vel]

if True:
    # generating grid
    # upper airfoil to look at flow oscillation
    x_lim = [-0.25, 1.25]
    z_lim = [0.025, 0.15]

    # near field
    # x_lim = [-0.25, 1.25]
    # z_lim = [-0.2, 0.2]

    # normal area around airfoil
    # x_lim = [-0.25, 1.25]
    # z_lim = [-0.5, 0.5]

    # captures far field
    # x_lim = [-1, 3]
    # z_lim = [-1, 1]

    nx, nz = 1000, 1000
    num_pts = nx*nz
    x_vec = np.linspace(x_lim[0], x_lim[1], num=nx)
    z_vec = np.linspace(z_lim[0], z_lim[1], num=nz)
    eval_points_mesh = np.zeros((nx, nz, 3))
    for i in range(nx):
        eval_points_mesh[i,:,0] = x_vec[i]
        eval_points_mesh[i,:,2] = z_vec

    eval_points_mesh_vec = eval_points_mesh.reshape((1, nx*nz, 3))
    total_vel_vec = np.zeros_like(eval_points_mesh_vec)

    # looping through grid points and running the simulator
    # for 100x100, takes less than a second
    # for 1000x1000, takes close to a minute and a half
    for i in range(num_pts):
        evaluation_pt = eval_points_mesh_vec[:,i,:]
        jax_sim[eval_pts] = evaluation_pt

        jax_sim.run()

        total_vel = jax_sim[eval_pt_velocity]
        total_vel_vec[:,i,:] = total_vel

    total_vel_grid = total_vel_vec.reshape((1,nx,nz,3))
    total_vel_norm = np.linalg.norm(total_vel_grid, axis=3) # total velocity norm
    V_inf_norm = np.linalg.norm(free_stream_vel) # free stream norm

    normalized_total_vel_grid = total_vel_grid/V_inf_norm
    normalized_total_vel_norm = total_vel_norm/V_inf_norm

    max_vel_norm = np.max(total_vel_norm)
    normalized_max_vel_norm = np.max(normalized_total_vel_norm)

    mesh_pts = mesh[0,:,int((ns-1)/2)]
    # bound_vortex_pts = mesh_dict['surface_0']['bound_vortex_mesh'][0,:,int((ns-1)/2)]

    X, Z = np.meshgrid(x_vec, z_vec)

    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max_vel_norm)
    fig, ax = plt.subplots()
    strm = ax.streamplot(X, Z, total_vel_grid[0,:,:,0].T, total_vel_grid[0,:,:,2].T, color=total_vel_norm[0,:,:].T, norm=norm, cmap='jet')
    ax.plot(mesh_pts[:,0], mesh_pts[:,2], 'k-o', linewidth=2)
    # ax.plot(bound_vortex_pts[:,0], bound_vortex_pts[:,2], 'm-o', linewidth=3)
    ax.set_xlim([x_vec[0], x_vec[-1]])
    ax.set_ylim([z_vec[0], z_vec[-1]])
    ax.set_xlabel('Normalized chord (x/c)')
    ax.set_ylabel('z')
    cbar = fig.colorbar(strm.lines)
    cbar.ax.set_xlabel(r'$|u|$', fontsize=18)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=normalized_max_vel_norm)
    fig, ax = plt.subplots()
    strm = ax.streamplot(X, Z, normalized_total_vel_grid[0,:,:,0].T, normalized_total_vel_grid[0,:,:,2].T, color=normalized_total_vel_norm[0,:,:].T, norm=norm, cmap='jet')
    ax.plot(mesh_pts[:,0], mesh_pts[:,2], 'k-o', linewidth=2)
    # ax.plot(bound_vortex_pts[:,0], bound_vortex_pts[:,2], 'm-o', linewidth=3)
    ax.set_xlim([x_vec[0], x_vec[-1]])
    ax.set_ylim([z_vec[0], z_vec[-1]])
    ax.set_xlabel('Normalized chord (x/c)')
    ax.set_ylabel('z')

    cbar = fig.colorbar(strm.lines)
    cbar.ax.set_xlabel(r'$\frac{|u|}{|U_\infty|}$', fontsize=18)

    plt.show()
