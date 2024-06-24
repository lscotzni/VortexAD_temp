import csdl_alpha as csdl

from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_strengths, compute_source_influence 
from VortexAD.core.panel_method.source_doublet.doublet_functions import compute_doublet_influence
from VortexAD.core.panel_method.source_doublet.wake_geometry import wake_geometry

from VortexAD.core.panel_method.source_doublet.free_wake_comp import free_wake_comp

def transient_solver(mesh_dict, wake_mesh_dict, num_nodes, nt, num_tot_panels, dt, free_wake=False):
    '''
    we solve for all AIC matrices here and sigma influence
    '''
    surface_names = list(mesh_dict.keys())
    num_surfaces = len(surface_names)

    sigma = compute_source_strengths(mesh_dict, surface_names, num_nodes, nt, num_tot_panels) # shape=(num_nodes, nt, num_surf_panels)

    AIC_sigma = csdl.Variable(shape=(num_nodes, nt, num_tot_panels, num_tot_panels), value=0.)
    AIC_mu = csdl.Variable(shape=(num_nodes, nt, num_tot_panels, num_tot_panels), value=0.)

    start_i, stop_i = 0, 0
    with csdl.namespace('Static AIC computation'):
        for i in range(num_surfaces):
            surf_i_name = surface_names[i]
            coll_point_i = mesh_dict[surf_i_name]['panel_center'] # evaluation point
            nc_i, ns_i = mesh_dict[surf_i_name]['nc'], mesh_dict[surf_i_name]['ns']
            num_panels_i = mesh_dict[surf_i_name]['num_panels']
            stop_i += num_panels_i

            start_j, stop_j = 0, 0
            for j in range(num_surfaces):
                surf_j_name = surface_names[j]
                nc_j, ns_j = mesh_dict[surf_j_name]['nc'], mesh_dict[surf_j_name]['ns']
                num_panels_j = mesh_dict[surf_j_name]['num_panels']
                stop_j += num_panels_j

                panel_corners_j = mesh_dict[surf_j_name]['panel_corners']
                panel_x_dir_j = mesh_dict[surf_j_name]['panel_x_dir']
                panel_y_dir_j = mesh_dict[surf_j_name]['panel_y_dir']
                panel_normal_j = mesh_dict[surf_j_name]['panel_normal']
                dpij_j = mesh_dict[surf_j_name]['dpij']
                dij_j = mesh_dict[surf_j_name]['dij']
                mij_j = mesh_dict[surf_j_name]['mij']

                num_interactions = num_panels_i*num_panels_j

                coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nt, nc_i-1, ns_i-1, num_panels_j, 4, 3), 'ijklm->ijklabm')
                coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, nt, num_interactions, 4, 3))

                panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklmn->ijaklmn')
                panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))

                panel_x_dir_j_exp = csdl.expand(panel_x_dir_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklm->ijaklbm')
                panel_x_dir_j_exp_vec = panel_x_dir_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))
                panel_y_dir_j_exp = csdl.expand(panel_y_dir_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklm->ijaklbm')
                panel_y_dir_j_exp_vec = panel_y_dir_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))
                panel_normal_j_exp = csdl.expand(panel_normal_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklm->ijaklbm')
                panel_normal_j_exp_vec = panel_normal_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))

                dpij_j_exp = csdl.expand(dpij_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 2), 'ijklmn->ijaklmn')
                dpij_j_exp_vec = dpij_j_exp.reshape((num_nodes, nt, num_interactions, 4, 2))
                dij_j_exp = csdl.expand(dij_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4), 'ijklm->ijaklm')
                dij_j_exp_vec = dij_j_exp.reshape((num_nodes, nt, num_interactions, 4))
                mij_j_exp = csdl.expand(mij_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4), 'ijklm->ijaklm')
                mij_j_exp_vec = mij_j_exp.reshape((num_nodes, nt, num_interactions, 4))

                dp = coll_point_i_exp_vec - panel_corners_j_exp_vec
                # NOTE: consider changing dx,dy,dz and store in just dk with a dimension for x,y,z
                sum_ind = len(dp.shape) - 1
                dx = csdl.sum(dp*panel_x_dir_j_exp_vec, axes=(sum_ind,))
                dy = csdl.sum(dp*panel_y_dir_j_exp_vec, axes=(sum_ind,))
                dz = csdl.sum(dp*panel_normal_j_exp_vec, axes=(sum_ind,))
                rk = (dx**2 + dy**2 + dz**2 + 1.e-12)**0.5
                ek = dx**2 + dz**2
                hk = dx*dy

                mij_list = [mij_j_exp_vec[:,:,:,ind] for ind in range(4)]
                dij_list = [dij_j_exp_vec[:,:,:,ind] for ind in range(4)]
                dpij_list = [[dpij_j_exp_vec[:,:,:,ind,0], dpij_j_exp_vec[:,:,:,ind,1]] for ind in range(4)]
                ek_list = [ek[:,:,:,ind] for ind in range(4)]
                hk_list = [hk[:,:,:,ind] for ind in range(4)]
                rk_list = [rk[:,:,:,ind] for ind in range(4)]
                dx_list = [dx[:,:,:,ind] for ind in range(4)]
                dy_list = [dy[:,:,:,ind] for ind in range(4)]
                dz_list = [dz[:,:,:,ind] for ind in range(4)]

                # source_influence_vec = compute_source_influence(dij_j_exp_vec, mij_j_exp_vec, dpij_j_exp_vec, dx, dy, dz, rk, ek, hk, mode='potential')
                source_influence_vec = compute_source_influence(dij_list, mij_list, dpij_list, dx_list, dy_list, dz_list, rk_list, ek_list, hk_list, mode='potential')
                source_influence = source_influence_vec.reshape((num_nodes, nt, num_panels_i, num_panels_j))
                AIC_sigma = AIC_sigma.set(csdl.slice[:,:,start_i:stop_i, start_j:stop_j], value=source_influence)

                # doublet_influence_vec = compute_doublet_influence(dpij_j_exp_vec, mij_j_exp_vec, ek, hk, rk, dx, dy, dz, mode='potential')
                doublet_influence_vec = compute_doublet_influence(dpij_list, mij_list, ek_list, hk_list, rk_list, dx_list, dy_list, dz_list, mode='potential')
                doublet_influence = doublet_influence_vec.reshape((num_nodes, nt, num_panels_i, num_panels_j))
                AIC_mu = AIC_mu.set(csdl.slice[:,:,start_i:stop_i, start_j:stop_j], value=doublet_influence)

                start_j += num_panels_j
            start_i += num_panels_i

    sigma_BC_influence = csdl.einsum(AIC_sigma, sigma, action='ijkl,ijk->ijl')
    # looping through the wake
    # we add part of this to the AIC_mu 
    num_wake_panels = 0
    num_first_wake_panels = 0
    for surface in surface_names:
        num_wake_panels += wake_mesh_dict[surface]['num_panels']
        num_first_wake_panels += wake_mesh_dict[surface]['ns'] - 1

    AIC_wake = csdl.Variable(shape=(num_nodes, nt, num_tot_panels, num_wake_panels - num_first_wake_panels), value=0.)
    AIC_mu_adjustment = csdl.Variable(shape=AIC_mu.shape, value=0.)
    AIC_mu_total = csdl.Variable(shape=AIC_mu.shape, value=0.)
    mu = csdl.Variable(shape=(num_nodes, nt, num_tot_panels), value=0.)
    mu_wake = csdl.Variable(shape=(num_nodes, nt, num_wake_panels), value=0.)
    mu_wake_minus_1 = csdl.Variable(shape=(num_nodes, nt, num_wake_panels - num_first_wake_panels), value=0.)
    
    with csdl.namespace('transient solver'):
        for t in csdl.frange(nt-1):
            start_mu, stop_mu = 0, 0
            start_mu_m1, stop_mu_m1 = 0, 0

            print(f'timestep {t}')
            start_i, stop_i = 0, 0
            for i in range(num_surfaces): # surface where BC is applied

                surf_i_name = surface_names[i]
                coll_point_i = mesh_dict[surf_i_name]['panel_center'][:,t,:,:,:] # evaluation point
                nc_i, ns_i = mesh_dict[surf_i_name]['nc'], mesh_dict[surf_i_name]['ns']
                num_panels_i = mesh_dict[surf_i_name]['num_panels']
                stop_i += num_panels_i

                start_j, stop_j = 0, 0
                start_j_surf, stop_j_surf = 0, 0
                for j in range(num_surfaces): # looping through wakes

                    surf_j_name = surface_names[j]
                    nc_j, ns_j = wake_mesh_dict[surf_j_name]['nc'], wake_mesh_dict[surf_j_name]['ns']
                    num_panels_j = wake_mesh_dict[surf_j_name]['num_panels']
                    stop_j += num_panels_j - (ns_j-1)

                    panel_corners_j = wake_mesh_dict[surf_j_name]['panel_corners'][:,t,:,:,:,:]
                    panel_x_dir_j = wake_mesh_dict[surf_j_name]['panel_x_dir'][:,t,:,:,:]
                    panel_y_dir_j = wake_mesh_dict[surf_j_name]['panel_y_dir'][:,t,:,:,:]
                    panel_normal_j = wake_mesh_dict[surf_j_name]['panel_normal'][:,t,:,:,:]
                    dpij_j = wake_mesh_dict[surf_j_name]['dpij'][:,t,:,:,:,:]
                    dij_j = wake_mesh_dict[surf_j_name]['dij'][:,t,:,:,:]
                    mij_j = wake_mesh_dict[surf_j_name]['mij'][:,t,:,:,:]

                    num_interactions = num_panels_i*num_panels_j

                    coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nc_i-1, ns_i-1, num_panels_j, 4, 3), 'jklm->jklabm')
                    coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, num_interactions, 4, 3))

                    panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklmn->jaklmn')
                    panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, num_interactions, 4, 3))

                    panel_x_dir_j_exp = csdl.expand(panel_x_dir_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
                    panel_x_dir_j_exp_vec = panel_x_dir_j_exp.reshape((num_nodes, num_interactions, 4, 3))
                    panel_y_dir_j_exp = csdl.expand(panel_y_dir_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
                    panel_y_dir_j_exp_vec = panel_y_dir_j_exp.reshape((num_nodes, num_interactions, 4, 3))
                    panel_normal_j_exp = csdl.expand(panel_normal_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
                    panel_normal_j_exp_vec = panel_normal_j_exp.reshape((num_nodes, num_interactions, 4, 3))


                    dpij_j_exp = csdl.expand(dpij_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 2), 'jklmn->jaklmn')
                    dpij_j_exp_vec = dpij_j_exp.reshape((num_nodes, num_interactions, 4, 2))
                    dij_j_exp = csdl.expand(dij_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
                    dij_j_exp_vec = dij_j_exp.reshape((num_nodes, num_interactions, 4))
                    mij_j_exp = csdl.expand(mij_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
                    mij_j_exp_vec = mij_j_exp.reshape((num_nodes, num_interactions, 4))

                    dp = coll_point_i_exp_vec - panel_corners_j_exp_vec
                    # NOTE: consider changing dx,dy,dz and store in just dk with a dimension for x,y,z
                    sum_ind = len(dp.shape) - 1
                    dx = csdl.sum(dp*panel_x_dir_j_exp_vec, axes=(sum_ind,))
                    dy = csdl.sum(dp*panel_y_dir_j_exp_vec, axes=(sum_ind,))
                    dz = csdl.sum(dp*panel_normal_j_exp_vec, axes=(sum_ind,))
                    rk = (dx**2 + dy**2 + dz**2 + 1.e-12)**0.5
                    ek = dx**2 + dz**2
                    hk = dx*dy

                    mij_list = [mij_j_exp_vec[:,:,ind] for ind in range(4)]
                    dpij_list = [[dpij_j_exp_vec[:,:,ind,0], dpij_j_exp_vec[:,:,ind,1]] for ind in range(4)]
                    ek_list = [ek[:,:,ind] for ind in range(4)]
                    hk_list = [hk[:,:,ind] for ind in range(4)]
                    rk_list = [rk[:,:,ind] for ind in range(4)]
                    dx_list = [dx[:,:,ind] for ind in range(4)]
                    dy_list = [dy[:,:,ind] for ind in range(4)]
                    dz_list = [dz[:,:,ind] for ind in range(4)]

                    # wake_doublet_influence_vec = compute_doublet_influence(dpij_j_exp_vec, mij_j_exp_vec, ek, hk, rk, dx, dy, dz, mode='potential')
                    wake_doublet_influence_vec = compute_doublet_influence(dpij_list, mij_list, ek_list, hk_list, rk_list, dx_list, dy_list, dz_list, mode='potential')

                    wake_doublet_influence = wake_doublet_influence_vec.reshape((num_nodes, num_panels_i, num_panels_j)) # GRID OF WAKE AIC (without kutta condition taken into account)

                    AIC_wake = AIC_wake.set(csdl.slice[:,t,start_i:stop_i, start_j:stop_j], value=wake_doublet_influence[:,:,(ns_j-1):])

                    # ADJUST AIC_mu HERE 
                    kutta_condition = wake_doublet_influence[:,:,:(ns_j-1)] # gets added to AIC_mu
                    # remainder of "wake_doublet_influence" is used in the AIC_wake matrix

                    nc_j_surf, ns_j_surf = mesh_dict[surf_j_name]['nc'], mesh_dict[surf_j_name]['ns']
                    num_panels_j_surf = mesh_dict[surf_j_name]['num_panels']
                    stop_j_surf += num_panels_j_surf

                    AIC_mu_adjustment = AIC_mu_adjustment.set(csdl.slice[:,t,start_i:stop_i, start_j_surf:start_j_surf+(ns_j-1)], value=-kutta_condition) # lower TE adjustment
                    AIC_mu_adjustment = AIC_mu_adjustment.set(csdl.slice[:,t,start_i:stop_i, (stop_j_surf-(ns_j-1)):stop_j_surf], value=kutta_condition) # upper TE adjustment
                    
                    AIC_mu = AIC_mu.set(csdl.slice[:,t,start_i:stop_i, start_j_surf:start_j_surf+(ns_j-1)], value=kutta_condition+AIC_mu[:,t,start_i:stop_i, start_j_surf:start_j_surf+(ns_j-1)])
                    AIC_mu = AIC_mu.set(csdl.slice[:,t,start_i:stop_i, (stop_j_surf-(ns_j-1)):stop_j_surf], value=kutta_condition+AIC_mu[:,t,start_i:stop_i, (stop_j_surf-(ns_j-1)):stop_j_surf])
                    1

                    start_j += num_panels_j - ns_j
                    start_j_surf += num_panels_j_surf

                start_i += num_panels_i

                # ADJUSTING MU FOR THE LINEAR SYSTEM SOLVE (sub variable with the first row of wake panels missing)
                num_surf_wake_panels = wake_mesh_dict[surf_i_name]['num_panels']
                num_surf_wake_first_panels = wake_mesh_dict[surf_i_name]['ns'] - 1

                stop_mu += num_surf_wake_panels
                stop_mu_m1 += num_surf_wake_panels - num_surf_wake_first_panels

                mu_wake_minus_1 = mu_wake_minus_1.set(csdl.slice[:,t,start_mu_m1:stop_mu_m1], value=mu_wake[:,t,(start_mu+num_surf_wake_first_panels):stop_mu])
                
                start_mu += num_surf_wake_panels
                start_mu_m1 += num_surf_wake_panels - num_surf_wake_first_panels
            
            # solving linear system:
            # # RHS
            # wake_influence = csdl.einsum(AIC_wake[:,t,:,:], mu_wake[:,t,:], action='ijk,ik->ij')
            # RHS = -sigma_BC_influence[:,t,:] - wake_influence
            # # RHS = -sigma_BC_influence[:,t,:]
            # # AIC_mu_total = AIC_mu_total.set(csdl.slice[:,t,:,:], value=AIC_mu[:,t,:,:]+AIC_mu_adjustment[:,t,:,:])
            # for nn in csdl.frange(num_nodes):
            #     # mu_timestep = csdl.solve_linear(AIC_mu_total[nn, t, :, :], RHS[nn,:])
            #     mu_timestep = csdl.solve_linear(AIC_mu[nn, t, :, :], RHS[nn,:])
            #     mu = mu.set(csdl.slice[nn,t,:], value=mu_timestep)
            
            # ============ ADD STEP HERE THAT SEGMENTS THE FIRST WAKE ROW FROM OTHERS ============
            
            for nn in csdl.frange(num_nodes):
                # RHS
                wake_influence = csdl.matvec(AIC_wake[nn,t,:,:], mu_wake_minus_1[nn,t,:])
                RHS = -sigma_BC_influence[nn,t,:] - wake_influence
                # RHS = -sigma_BC_influence[:,t,:]
                AIC_mu_total = AIC_mu_total.set(csdl.slice[:,t,:,:], value=AIC_mu[:,t,:,:]+AIC_mu_adjustment[:,t,:,:])
                mu_timestep = csdl.solve_linear(AIC_mu_total[nn, t, :, :], RHS)
                # mu_timestep = csdl.solve_linear(AIC_mu[nn, t, :, :], RHS)
                mu = mu.set(csdl.slice[nn,t,:], value=mu_timestep)

            if t == nt-1:
                continue
            if free_wake:
                # induced_vel = csdl.Variable(shape=(num_nodes, nt, num_wake_panels, 3)) # CHANGE IN THE FUTURE TO BE THE NODES
                induced_vel = free_wake_comp(num_nodes, t, mesh_dict, mu, sigma, wake_mesh_dict, mu_wake)
                # print(induced_vel.shape)

            # update mu and position of mesh
            start_i_s, stop_i_s = 0, 0
            start_i_w, stop_i_w = 0, 0
            for i in range(num_surfaces):
                surface_name = surface_names[i]
                nc_s, ns_s =  mesh_dict[surface_name]['nc'], mesh_dict[surface_name]['ns']
                num_panels_s = mesh_dict[surface_name]['num_panels']
                stop_i_s += num_panels_s

                nc_w, ns_w =  wake_mesh_dict[surface_name]['nc'], wake_mesh_dict[surface_name]['ns']
                num_panels_w = wake_mesh_dict[surface_name]['num_panels']
                # stop_i_w += num_panels_w - (ns_w-1)
                stop_i_w += num_panels_w

                # shift wake doublet strengths
                mu_surf_grid = mu[:,t, start_i_s:stop_i_s].reshape((num_nodes, nc_s-1, ns_s-1))
                mu_wake_first_row = mu_surf_grid[:,-1,:] - mu_surf_grid[:,0,:] # mu_upper - mu_lower @ TE

                mu_wake_grid = mu_wake[:,t,start_i_w:stop_i_w].reshape((num_nodes, nc_w-1, ns_w-1))
                mu_wake_grid_next = csdl.Variable(shape=mu_wake_grid.shape, value=0.)
                mu_wake_grid_next = mu_wake_grid_next.set(csdl.slice[:,0,:], value=mu_wake_first_row)
                mu_wake_grid_next = mu_wake_grid_next.set(csdl.slice[:,1:,:], value=mu_wake_grid[:,0:-1])
                # mu_wake_grid_next = mu_wake_grid_next.set(csdl.slice[:,1:,:], value=mu_wake_grid[:,0:-1])

                mu_wake = mu_wake.set(csdl.slice[:,t+1,start_i_w:stop_i_w], value=mu_wake_grid_next.reshape((num_nodes, (nc_w-1)*(ns_w-1))))
                # mu_wake = mu_wake.set(csdl.slice[:,t+1,start_i_w+(ns_w-1):start_i_w+(ns_w-1)*(t+1)], value=mu_wake[:,t,start_i_w:start_i_w+(ns_w-1)*t])

                # propagate wake
                wake_mesh = wake_mesh_dict[surface_name]['mesh']
                wake_velocity = wake_mesh_dict[surface_name]['wake_nodal_velocity']
                
                wake_velocity_timestep = wake_velocity[:,t,:,:,:]

                if free_wake:
                    total_vel = wake_velocity_timestep + induced_vel[:,:].reshape((num_nodes, nc_w, ns_w, 3))
                else:
                    total_vel = wake_velocity_timestep
                dx = total_vel*dt
                # wake_mesh[:,:,0,:,:] is already initialized at the TE
                wake_mesh = wake_mesh.set(csdl.slice[:,t+1,2:,:,:], value=dx[:,1:-1,:,:] + wake_mesh[:,t,1:-1,:,:])
                
                wake_velocity = wake_velocity.set(csdl.slice[:,t+1,:2,:,:], value=wake_velocity[:,t,:2,:,:])
                wake_velocity = wake_velocity.set(csdl.slice[:,t+1,2:,:,:], value=wake_velocity[:,t,1:-1,:,:])

                wake_mesh_dict[surface_name]['mesh'] = wake_mesh
                wake_mesh_dict[surface_name]['wake_nodal_velocity'] = wake_velocity

                wake_mesh_dict[surface_name] = wake_geometry(surf_wake_mesh_dict=wake_mesh_dict[surface_name], time_ind=t+1)
            1

        

    return mu, sigma, mu_wake