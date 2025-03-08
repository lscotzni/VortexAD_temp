import numpy as np 
import matplotlib.pyplot as plt 
import csdl_alpha as csdl

from VortexAD.core.vlm.velocity_computations import compute_induced_velocity

import pickle

def flow_field_superposition(eval_pts, mesh_dict, mesh_velocity, gamma):
    '''
    eval_pts is the input array of evaluation points; this should be a numpy array 
    of shape (num_nodes, num_eval_pts, 3)
        - if not, it will be reshaped
    '''
    num_nodes = 1
    # eval_pts_orig = csdl.Variable(value=eval_pts)
    eval_pts_orig = eval_pts
    eval_pts_shape = eval_pts.shape
    if len(eval_pts_shape) == 1: # means only one evaluation point
        num_eval_pts = 1
        eval_pts = csdl.reshape(eval_pts_orig, (num_nodes, 1, 3)) 
    elif len(eval_pts_shape) == 2:
        num_eval_pts = eval_pts_shape[0]
        eval_pts = csdl.reshape(eval_pts_orig, (num_nodes, num_eval_pts, 3))
    elif len(eval_pts_shape) == 3:
        num_eval_pts = eval_pts_shape[1]
        eval_pts = eval_pts_orig

    # compute AIC
    surface_names = list(mesh_dict.keys())
    num_surfaces = len(surface_names)

    num_total_panels = 0
    for key in mesh_dict.keys():
        ns, nc = mesh_dict[key]['ns'], mesh_dict[key]['nc']
        num_total_panels += (ns-1)*(nc-1)

    AIC = csdl.Variable(shape=(num_nodes, num_eval_pts, num_total_panels, 3), value=0.)
    start, stop = 0, 0
    for j in range(num_surfaces):
        surface_name_j = surface_names[j]
        ns_j, nc_j = mesh_dict[surface_name_j]['ns'], mesh_dict[surface_name_j]['nc']
        np_surf_j = (ns_j-1)*(nc_j-1)

        bound_vortex_mesh_j = mesh_dict[surface_name_j]['bound_vortex_mesh']

        # VECTORIZE AND EXPAND EVERYTHING TO SHAPE (num_nodes, np_surf_j*np_surf_i) (+ (3,) if needed)
        num_interactions = num_eval_pts*np_surf_j

        p1_bd_grid = bound_vortex_mesh_j[:, :-1, :-1, :]
        p2_bd_grid = bound_vortex_mesh_j[:, :-1, 1:, :]
        p3_bd_grid = bound_vortex_mesh_j[:, 1:, 1:, :]
        p4_bd_grid = bound_vortex_mesh_j[:, 1:, :-1, :]

        p1_bd = csdl.expand(p1_bd_grid, (num_nodes, num_eval_pts, nc_j-1, ns_j-1, 3), 'ijkl->iajkl')
        p2_bd = csdl.expand(p2_bd_grid, (num_nodes, num_eval_pts, nc_j-1, ns_j-1, 3), 'ijkl->iajkl')
        p3_bd = csdl.expand(p3_bd_grid, (num_nodes, num_eval_pts, nc_j-1, ns_j-1, 3), 'ijkl->iajkl')
        p4_bd = csdl.expand(p4_bd_grid, (num_nodes, num_eval_pts, nc_j-1, ns_j-1, 3), 'ijkl->iajkl')

        interaction_shape = (num_nodes, num_interactions, 3)

        # reshape starts with the right-most dimensions. So, this vectorizes panel 0 np_surf_i times, then panel 1, ...
        p1_bd_vec = p1_bd.reshape(interaction_shape)
        p2_bd_vec = p2_bd.reshape(interaction_shape)
        p3_bd_vec = p3_bd.reshape(interaction_shape)
        p4_bd_vec = p4_bd.reshape(interaction_shape)

        eval_pt_i_exp = csdl.expand(eval_pts, (num_nodes,  num_eval_pts, np_surf_j, 3), 'ijk->ijak')
        eval_pt_i_exp_vec = eval_pt_i_exp.reshape((num_nodes, num_interactions, 3))

        v_i_12 = compute_induced_velocity(p1_bd_vec, p2_bd_vec, eval_pt_i_exp_vec, vc=1.e-6)
        v_i_23 = compute_induced_velocity(p2_bd_vec, p3_bd_vec, eval_pt_i_exp_vec, vc=1.e-6)
        v_i_34 = compute_induced_velocity(p3_bd_vec, p4_bd_vec, eval_pt_i_exp_vec, vc=1.e-6)
        v_i_41 = compute_induced_velocity(p4_bd_vec, p1_bd_vec, eval_pt_i_exp_vec, vc=1.e-6)

        v_induced = v_i_12 + v_i_23 + v_i_34 + v_i_41
        v_induced_grid = v_induced.reshape((num_nodes, num_eval_pts, np_surf_j, 3))

        wake_vortex_mesh_j  = mesh_dict[surface_name_j]['wake_vortex_mesh']
        nc_w_j, ns_w_j = wake_vortex_mesh_j.shape[1], wake_vortex_mesh_j.shape[2]
        np_wake_j = (nc_w_j-1)*(ns_w_j-1)

        num_wake_interactions = num_eval_pts*np_wake_j
        wake_interaction_shape = (num_nodes, num_wake_interactions, 3)

        p1_w_grid = wake_vortex_mesh_j[:, :-1, :-1, :]
        p2_w_grid = wake_vortex_mesh_j[:, :-1, 1:, :]
        p3_w_grid = wake_vortex_mesh_j[:, 1:, 1:, :]
        p4_w_grid = wake_vortex_mesh_j[:, 1:, :-1, :]

        p1_w = csdl.expand(p1_w_grid, (num_nodes, num_eval_pts, nc_w_j-1, ns_w_j-1, 3), 'ijkl->iajkl')
        p2_w = csdl.expand(p2_w_grid, (num_nodes, num_eval_pts, nc_w_j-1, ns_w_j-1, 3), 'ijkl->iajkl')
        p3_w = csdl.expand(p3_w_grid, (num_nodes, num_eval_pts, nc_w_j-1, ns_w_j-1, 3), 'ijkl->iajkl')
        p4_w = csdl.expand(p4_w_grid, (num_nodes, num_eval_pts, nc_w_j-1, ns_w_j-1, 3), 'ijkl->iajkl')

        p1_w_vec = p1_w.reshape(wake_interaction_shape)
        p2_w_vec = p2_w.reshape(wake_interaction_shape)
        p3_w_vec = p3_w.reshape(wake_interaction_shape)
        p4_w_vec = p4_w.reshape(wake_interaction_shape)

        eval_pt_i_wake_exp = csdl.expand(eval_pts, (num_nodes, num_eval_pts, np_wake_j, 3), 'ijk->ijak')
        # eval_pt_i_wake_exp = csdl.Variable(shape=(num_nodes, nc_i-1, ns_i-1, np_wake_j, 3), value=0.)
        # for k in csdl.frange((np_wake_j)):
        #     eval_pt_i_wake_exp = eval_pt_i_wake_exp.set(csdl.slice[:,:,:,k,:], value=eval_pt_i)

        eval_pt_i_wake_exp_vec = eval_pt_i_wake_exp.reshape(wake_interaction_shape)

        v_i_12_w = compute_induced_velocity(p1_w_vec, p2_w_vec, eval_pt_i_wake_exp_vec, vc=1.e-6)
        v_i_23_w = compute_induced_velocity(p2_w_vec, p3_w_vec, eval_pt_i_wake_exp_vec, vc=1.e-6)
        v_i_34_w = compute_induced_velocity(p3_w_vec, p4_w_vec, eval_pt_i_wake_exp_vec, vc=1.e-6)
        v_i_41_w = compute_induced_velocity(p4_w_vec, p1_w_vec, eval_pt_i_wake_exp_vec, vc=1.e-6)

        v_induced_wake = v_i_12_w + v_i_23_w + v_i_34_w + v_i_41_w

        v_induced_wake_grid = v_induced_wake.reshape((num_nodes, num_eval_pts, np_wake_j, 3))
        wake_influence_grid = csdl.Variable(shape=v_induced_grid.shape, value=0.)
        stop += np_surf_j
        start_wake_ind = stop - np_wake_j

        # wake_influence_grid = wake_influence_grid.set(csdl.slice[:,-(stop_panel_counter_j-start_wake_ind):,start_panel_counter:stop_panel_counter,:], value=v_induced_wake_grid)
        wake_influence_grid = wake_influence_grid.set(csdl.slice[:,:,-(stop-start_wake_ind):,:], value=v_induced_wake_grid)

        induced_vel = v_induced_grid + wake_influence_grid
        # induced_vel = v_induced_grid + v_induced_wake_grid
        AIC = AIC.set(csdl.slice[:, :, start:stop, :], value=induced_vel)


    eval_pt_induced_vel = csdl.einsum(
        AIC,
        gamma,
        action='ijkl,ik->ijl'
    )
    freestream_velocity = csdl.average(mesh_velocity, axes=(1,2))
    freestream_vel_exp = csdl.expand(freestream_velocity, eval_pt_induced_vel.shape,'ij->iaj')
    total_velocity = eval_pt_induced_vel + freestream_vel_exp

    return total_velocity

file_name = 'superposition_data.pickle'
with open(file_name, 'rb') as file:
    data = pickle.load(file)

mesh_dict = data['mesh_dict']
mesh_velocity = data['velocity'] * -1 # bc of reference frame 
alpha_deg = data['alpha']
gamma = data['gamma']

recorder = csdl.Recorder(inline=False)
recorder.start()
mesh = csdl.Variable(value=mesh_dict['surface_0']['mesh'])
ns = mesh_dict['surface_0']['ns']
nc = mesh_dict['surface_0']['nc']

eval_pts = csdl.Variable(value=np.zeros((1,3))) # single point evaluation
num_eval_pts = 10
eval_pts = csdl.Variable(value=np.zeros((num_eval_pts,3))) # vectorized multi-point evaluation

total_vel_csdl = flow_field_superposition(
    eval_pts=eval_pts,
    mesh_dict=mesh_dict,
    mesh_velocity=mesh_velocity,
    gamma=gamma
)

recorder.stop()

jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[mesh, eval_pts],
    additional_outputs=[total_vel_csdl]
)

# I RECOMMEND SETTING UP THE VALUES FOR eval_pts HERE

jax_sim.run()
total_vel = jax_sim[total_vel_csdl]

# ==== STREAMLINE PLOT ====
if False:
    # generating grid
    x_lim = [-0.5, 1.5]
    z_lim = [-0.25, 0.25]
    nx, nz = 100, 100
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

        total_vel = jax_sim[total_vel_csdl]
        total_vel_vec[:,i,:] = total_vel

    total_vel_grid = total_vel_vec.reshape((1,nx,nz,3))
    total_vel_norm = np.linalg.norm(total_vel_grid, axis=3) # total velocity norm
    V_inf_norm = np.linalg.norm(mesh_velocity[0,0,0,:]) # free stream norm

    mesh_pts = mesh_dict['surface_0']['mesh'][0,:,int((ns-1)/2)]
    bound_vortex_pts = mesh_dict['surface_0']['bound_vortex_mesh'][0,:,int((ns-1)/2)]

    X, Z = np.meshgrid(x_vec, z_vec)

    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=0, vmax=V_inf_norm*1.5)
    fig, ax = plt.subplots()
    strm = ax.streamplot(X, Z, total_vel_grid[0,:,:,0].T, total_vel_grid[0,:,:,2].T, color=total_vel_norm[0,:,:].T, norm=norm, cmap='jet')
    ax.plot(mesh_pts[:,0], mesh_pts[:,2], 'k-o', linewidth=3)
    ax.plot(bound_vortex_pts[:,0], bound_vortex_pts[:,2], 'm-o', linewidth=3)
    ax.set_xlim([x_vec[0], x_vec[-1]])
    ax.set_ylim([z_vec[0], z_vec[-1]])
    ax.set_xlabel('Normalized chord (x/c)')
    ax.set_ylabel('z')

    cbar = fig.colorbar(strm.lines)
    cbar.ax.set_xlabel(r'$\frac{|u|}{|U_\infty|}$', fontsize=18)

    plt.show()

