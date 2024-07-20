import csdl_alpha as csdl 

from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_strengths, compute_source_influence
from VortexAD.core.panel_method.source_doublet.doublet_functions import compute_doublet_influence
from VortexAD.core.panel_method.source_doublet.wake_geometry import wake_geometry

from VortexAD.core.panel_method.source_doublet.free_wake_comp import free_wake_comp

def unstructured_transient_solver(mesh_dict, wake_mesh_dict, num_nodes, nt, num_tot_panels, dt, free_wake=False):
    sigma = compute_source_strengths(mesh_dict, num_nodes, nt, num_tot_panels, mesh_mode='unstructured')
    AIC_mu, AIC_sigma = static_AIC_computation(mesh_dict, num_nodes, nt, num_tot_panels)

    sigma_BC_influence = csdl.einsum(AIC_sigma, sigma, action='ijlk,ijk->ijl')

    upper_TE_cell_ind = mesh_dict['upper_TE_cells']
    lower_TE_cell_ind = mesh_dict['lower_TE_cells']
    num_first_wake_panels = len(upper_TE_cell_ind)
    num_wake_panels = wake_mesh_dict['num_panels']

    # initializing transient AIC matrix adjustments
    AIC_wake = csdl.Variable(shape=(num_nodes, nt, num_tot_panels, num_wake_panels-num_first_wake_panels), value=0.)
    AIC_mu_adjustment = csdl.Variable(shape=AIC_mu.shape, value=0.)
    AIC_mu_total = csdl.Variable(shape=AIC_mu.shape, value=0.)

    # initializing outputs
    mu = csdl.Variable(shape=(num_nodes, nt, num_tot_panels), value=0.)
    mu_wake = csdl.Variable(shape=(num_nodes, nt, num_wake_panels), value=0.)
    mu_wake_minus_1 = csdl.Variable(shape=(num_nodes, nt, num_wake_panels-num_first_wake_panels), value=0.)

    for t in csdl.frange(nt-1):
        # compute influence from wake here
        coll_point = mesh_dict['panel_center'][:,t,:,:] # (nn, num_tot_panels, 3)

        # setting up values for the wake
        # NOTE: wake mesh uses structured 4-sided panels
        panel_corners_w = wake_mesh_dict['panel_corners'][:,t,:,:,:,:] # (nn, nc_w, ns_w, 4, 3)
        panel_x_dir_w = wake_mesh_dict['panel_x_dir'][:,t,:,:,:] # (nn, nc_w, ns_w, 3)
        panel_y_dir_w = wake_mesh_dict['panel_y_dir'][:,t,:,:,:] # (nn, nc_w, ns_w, 3)
        panel_normal_w = wake_mesh_dict['panel_normal'][:,t,:,:,:] # (nn, nc_w, ns_w, 3)
        dpij_w = wake_mesh_dict['dpij'][:,t,:,:,:,:] # (nn, nc_w, ns_w, 4, 2)
        dij_w = wake_mesh_dict['dij'][:,t,:,:,:] # (nn, nc_w, ns_w, 4)

        nc_w, ns_w = panel_corners_w.shape[1], panel_corners_w.shape[2]

        # target expansion and vectorization shapes
        num_wake_interactions = num_tot_panels*num_wake_panels
        expanded_shape = (num_nodes, num_tot_panels, nc_w, ns_w, 4, 3)
        vectorized_shape = (num_nodes, num_wake_interactions, 4, 3)

        # expanding and vectorizing terms
        coll_point_exp = csdl.expand(coll_point, (num_nodes, num_tot_panels, num_wake_panels, 4, 3), 'ijk->ijabk')
        coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

        panel_corners_w_exp = csdl.expand(panel_corners_w, expanded_shape, 'ijklm->iajklm')
        panel_corners_w_exp_vec = panel_corners_w_exp.reshape(vectorized_shape)

        panel_x_dir_w_exp = csdl.expand(panel_x_dir_w, expanded_shape, 'ijkl->iajkbl')
        panel_x_dir_w_exp_vec = panel_x_dir_w_exp.reshape(vectorized_shape)
        panel_y_dir_w_exp = csdl.expand(panel_y_dir_w, expanded_shape, 'ijkl->iajkbl')
        panel_y_dir_w_exp_vec = panel_y_dir_w_exp.reshape(vectorized_shape)
        panel_normal_w_exp = csdl.expand(panel_normal_w, expanded_shape, 'ijkl->iajkbl')
        panel_normal_w_exp_vec = panel_normal_w_exp.reshape(vectorized_shape)

        dpij_w_exp = csdl.expand(dpij_w, expanded_shape[:-1] + (2,), 'ijklm->iajklm')
        dpij_w_exp_vec = dpij_w_exp.reshape(vectorized_shape[:-1] + (2,))
        dij_w_exp = csdl.expand(dij_w, expanded_shape[:-1], 'ijkl->iajkl')
        dij_w_exp_vec = dij_w_exp.reshape(vectorized_shape[:-1])

        dp = coll_point_exp_vec - panel_corners_w_exp_vec
        sum_ind = len(dp.shape) - 1
        dx = csdl.sum(dp*panel_x_dir_w_exp_vec, axes=(sum_ind,))
        dy = csdl.sum(dp*panel_y_dir_w_exp_vec, axes=(sum_ind,))
        dz = csdl.sum(dp*panel_normal_w_exp_vec, axes=(sum_ind,))
        rk = (dx**2 + dy**2 + dz**2 + 1.e-12)**0.5
        ek = dx**2 + dz**2
        hk = dx*dy

        # mij_list = [mij_exp_vec[:,:,ind] for ind in range(4)]
        dij_list = [dij_w_exp_vec[:,:,ind] for ind in range(4)]
        dpij_list = [[dpij_w_exp_vec[:,:,ind,0], dpij_w_exp_vec[:,:,ind,1]] for ind in range(4)]
        ek_list = [ek[:,:,ind] for ind in range(4)]
        hk_list = [hk[:,:,ind] for ind in range(4)]
        rk_list = [rk[:,:,ind] for ind in range(4)]
        dx_list = [dx[:,:,ind] for ind in range(4)]
        dy_list = [dy[:,:,ind] for ind in range(4)]
        dz_list = [dz[:,:,ind] for ind in range(4)]

        wake_doublet_influence_vec = compute_doublet_influence(
            dpij_list, 
            0, # we don't use this input for now 
            ek_list, 
            hk_list, 
            rk_list, 
            dx_list, 
            dy_list, 
            dz_list, 
            mode='potential'
        )
        wake_doublet_influence = wake_doublet_influence_vec.reshape((num_nodes, num_tot_panels, num_wake_panels))
        AIC_wake = AIC_wake.set(csdl.slice[:,t,:,:], value=wake_doublet_influence[:,:,num_first_wake_panels:]) # NOTE THIS ONLY WORKS FOR SINGLE SURFACE AT THIS POINT

        kutta_condition = wake_doublet_influence[:,:,:num_first_wake_panels]

        kc_reshaped = kutta_condition.reshape((len(lower_TE_cell_ind), num_nodes, num_tot_panels))

        AIC_mu_adjustment = AIC_mu_adjustment.set(csdl.slice[:,t,:,list(lower_TE_cell_ind)], value=-kc_reshaped)
        AIC_mu_adjustment = AIC_mu_adjustment.set(csdl.slice[:,t,:,list(upper_TE_cell_ind)], value=kc_reshaped)

        mu_wake_minus_1 = mu_wake_minus_1.set(csdl.slice[:,t,:], value=mu_wake[:,t,num_first_wake_panels:])

        # solving linear system
        # AIC_mu_total = AIC_mu_total.set(csdl.slice[:,t,:,:], value=AIC_mu[:,t,:,:]+AIC_mu_adjustment[:,t,:,:])
        AIC_mu_total = AIC_mu_total.set(csdl.slice[:,t,:,:], value=AIC_mu[:,t,:,:])
        for nn in csdl.frange(num_nodes):
            wake_influence = csdl.matvec(AIC_wake[nn,t,:,:], mu_wake_minus_1[nn,t,:])
            # RHS = -sigma_BC_influence[nn,t,:] - wake_influence
            RHS = -sigma_BC_influence[nn,t,:]
            mu_timestep = csdl.solve_linear(AIC_mu_total[nn,t,:,:], RHS)
            mu = mu.set(csdl.slice[nn,t,:], value=mu_timestep)
        
        if free_wake:
            induced_vel = free_wake_comp(num_nodes, t, mesh_dict, mu, sigma, wake_mesh_dict, mu_wake)

        # propagating doublet strengths into the wake
        mu_upper_TE, mu_lower_TE = mu[:,t,list(upper_TE_cell_ind)], mu[:,t,list(lower_TE_cell_ind)]
        mu_wake_first_row = mu_upper_TE - mu_lower_TE

        mu_wake_grid = mu_wake[:,t,:].reshape((num_nodes,nc_w,ns_w))
        mu_wake_grid_next = csdl.Variable(shape=mu_wake_grid.shape, value=0.)
        mu_wake_grid_next = mu_wake_grid_next.set(csdl.slice[:,0,:], value=mu_wake_first_row)
        mu_wake_grid_next = mu_wake_grid_next.set(csdl.slice[:,1:,:], value=mu_wake_grid[:,0:-1])

        mu_wake = mu_wake.set(csdl.slice[:,t+1,:], value=mu_wake_grid_next.reshape((num_nodes, num_wake_panels)))

        # propagating wake mesh
        wake_mesh = wake_mesh_dict['mesh']
        wake_velocity = wake_mesh_dict['wake_nodal_velocity']

        wake_velocity_timestep = wake_velocity[:,t,:,:,:]

        if free_wake:
            total_vel = wake_velocity_timestep - induced_vel[:,:].reshape((num_nodes, nc_w, ns_w, 3))
        else:
            total_vel = wake_velocity_timestep
        dx = total_vel*dt

        wake_mesh = wake_mesh.set(csdl.slice[:,t+1,2:,:,:], value=dx[:,1:-1,:,:] + wake_mesh[:,t,1:-1,:,:])

        wake_velocity = wake_velocity.set(csdl.slice[:,t+1,:2,:,:], value=wake_velocity[:,t,:2,:,:])
        wake_velocity = wake_velocity.set(csdl.slice[:,t+1,2:,:,:], value=wake_velocity[:,t,1:-1,:,:])

        wake_mesh_dict['mesh'] = wake_mesh
        wake_mesh_dict['wake_nodal_velocity'] = wake_velocity

        wake_mesh_dict = wake_geometry(wake_mesh_dict, time_ind=t+1)

    return mu, sigma, mu_wake

def static_AIC_computation(mesh_dict, num_nodes, nt, num_tot_panels):
    AIC_sigma = csdl.Variable(shape=(num_nodes, nt, num_tot_panels, num_tot_panels), value=0.)
    AIC_mu = csdl.Variable(shape=(num_nodes, nt, num_tot_panels, num_tot_panels), value=0.)

    '''
    NOTE: WE ARE NOT LOOPING OVER SURFACES FOR NOW
    - later, we will likely need a loop of some sorts (maybe not, we'll see)
    '''

    coll_point = mesh_dict['panel_center'] # (nn, nt, num_tot_panels, 3)
    panel_corners = mesh_dict['panel_corners'] # (nn, nt, num_tot_panels, 3, 3) 
    panel_x_dir = mesh_dict['panel_x_dir'] # (nn, nt, num_tot_panels, 3)
    panel_y_dir = mesh_dict['panel_y_dir'] # (nn, nt, num_tot_panels, 3)
    panel_normal = mesh_dict['panel_normal'] # (nn, nt, num_tot_panels, 3)
    dpij = mesh_dict['dpij'] # (nn, nt, num_tot_panels, 3, 2)
    dij = mesh_dict['dij'] # (nn, nt, num_tot_panels, 3)

    num_interactions = num_tot_panels**2
    expanded_shape = (num_nodes, nt, num_tot_panels, num_tot_panels, 3, 3)
    vectorized_shape = (num_nodes, nt, num_interactions, 3, 3)

    # expanding collocation points (where boundary condition is applied, the "i-th" expansion-vectorization)
    coll_point_exp = csdl.expand(coll_point, expanded_shape, 'ijkl->ijkabl')
    coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

    # expanding the panel terms used to compute influences AT the collocation points
    # -> the "j-th" expansion-vectorization
    panel_corners_exp = csdl.expand(panel_corners, expanded_shape, 'ijklm->ijaklm')
    panel_corners_exp_vec = panel_corners_exp.reshape(vectorized_shape)

    panel_x_dir_exp = csdl.expand(panel_x_dir, expanded_shape, 'ijkl->ijakbl')
    panel_x_dir_exp_vec = panel_x_dir_exp.reshape(vectorized_shape)
    panel_y_dir_exp = csdl.expand(panel_y_dir, expanded_shape, 'ijkl->ijakbl')
    panel_y_dir_exp_vec = panel_y_dir_exp.reshape(vectorized_shape)
    panel_normal_exp = csdl.expand(panel_normal, expanded_shape, 'ijkl->ijakbl')
    panel_normal_exp_vec = panel_normal_exp.reshape(vectorized_shape)

    dpij_exp = csdl.expand(dpij, expanded_shape[:-1] + (2,), 'ijklm->ijaklm')
    dpij_exp_vec = dpij_exp.reshape(vectorized_shape[:-1] + (2,))
    dij_exp = csdl.expand(dij, expanded_shape[:-1], 'ijkl->ijakl')
    dij_exp_vec = dij_exp.reshape(vectorized_shape[:-1])

    dp = coll_point_exp_vec - panel_corners_exp_vec
    sum_ind = len(dp.shape) - 1
    dx = csdl.sum(dp*panel_x_dir_exp_vec, axes=(sum_ind,))
    dy = csdl.sum(dp*panel_y_dir_exp_vec, axes=(sum_ind,))
    dz = csdl.sum(dp*panel_normal_exp_vec, axes=(sum_ind,))
    rk = (dx**2 + dy**2 + dz**2 + 1.e-12)**0.5
    ek = dx**2 + dz**2
    hk = dx*dy

    # mij_list = [mij_exp_vec[:,:,:,ind] for ind in range(4)]
    dij_list = [dij_exp_vec[:,:,:,ind] for ind in range(3)]
    dpij_list = [[dpij_exp_vec[:,:,:,ind,0], dpij_exp_vec[:,:,:,ind,1]] for ind in range(3)]
    ek_list = [ek[:,:,:,ind] for ind in range(3)]
    hk_list = [hk[:,:,:,ind] for ind in range(3)]
    rk_list = [rk[:,:,:,ind] for ind in range(3)]
    dx_list = [dx[:,:,:,ind] for ind in range(3)]
    dy_list = [dy[:,:,:,ind] for ind in range(3)]
    dz_list = [dz[:,:,:,ind] for ind in range(3)]

    source_influence_vec = compute_source_influence(
        dij_list,
        0, # we don't use this input for now
        dpij_list,
        dx_list,
        dy_list,
        dz_list,
        rk_list,
        ek_list,
        hk_list,
        mode='potential'
    )
    source_influence = source_influence_vec.reshape((num_nodes, nt, num_tot_panels, num_tot_panels))

    doublet_influence_vec = compute_doublet_influence(
        dpij_list, 
        0, # we don't use this input for now 
        ek_list, 
        hk_list, 
        rk_list, 
        dx_list, 
        dy_list, 
        dz_list, 
        mode='potential'
    )
    doublet_influence = doublet_influence_vec.reshape((num_nodes, nt, num_tot_panels, num_tot_panels))

    AIC_sigma = source_influence
    AIC_mu = doublet_influence

    return AIC_mu, AIC_sigma