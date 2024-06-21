import csdl_alpha as csdl

from VortexAD.core.panel_method.vortex_ring.vortex_line_functions import compute_vortex_line_ind_vel
from VortexAD.core.panel_method.vortex_ring.wake_geometry import wake_geometry

from VortexAD.core.panel_method.vortex_ring.free_wake_comp import free_wake_comp

from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_strengths

def transient_solver(mesh_dict, wake_mesh_dict, num_nodes, nt, num_tot_panels, dt, free_wake=False):
    '''
    we solve for all AIC matrices here and sigma influence
    '''
    surface_names = list(mesh_dict.keys())
    num_surfaces = len(surface_names)

    # NOTE: WE NEED THE NORMAL VELOCITY COMPONENT, WHICH IS GIVEN BY THE SOURCE STRENGTHS
    no_penetration_cond = compute_source_strengths(mesh_dict, surface_names, num_nodes, nt, num_tot_panels) # shape=(num_nodes, nt, num_surf_panels)
    
    AIC_gamma = csdl.Variable(shape=(num_nodes, nt, num_tot_panels, num_tot_panels), value=0.)

    start_i, stop_i = 0, 0
    with csdl.namespace('Static AIC computation'):
        for i in range(num_surfaces):
            surf_i_name = surface_names[i]
            coll_point_i = mesh_dict[surf_i_name]['panel_center'] # evaluation point
            panel_normal_i = mesh_dict[surf_i_name]['panel_normal'] # normal of collocation point
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
                panel_normal_j = mesh_dict[surf_j_name]['panel_normal']

                num_interactions = num_panels_i*num_panels_j

                # ============ EXPANDING THE TERMS FOR SURFACE i ============
                coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nt, nc_i-1, ns_i-1, num_panels_j, 4, 3), 'ijklm->ijklabm')
                coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, nt, num_interactions, 4, 3))

                panel_normal_i_exp = csdl.expand(panel_normal_i, (num_nodes, nt, nc_i-1, ns_i-1, num_panels_j, 4, 3), 'ijklm->ijklabm')
                panel_normal_i_exp_vec = panel_normal_i_exp.reshape((num_nodes, nt, num_interactions, 4, 3))

                # ============ EXPANDING THE TERMS FOR SURFACE j ============
                panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklmn->ijaklmn')
                panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))

                # panel_normal_j_exp = csdl.expand(panel_normal_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklm->ijaklbm')
                # panel_normal_j_exp_vec = panel_normal_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))

                ind_vel_12 = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,:,0,:], panel_corners_j_exp_vec[:,:,:,1,:], coll_point_i_exp_vec[:,:,:,0,:], mode='surface') # dim with 0 is the same so index doesn't matter
                ind_vel_23 = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,:,1,:], panel_corners_j_exp_vec[:,:,:,2,:], coll_point_i_exp_vec[:,:,:,0,:], mode='surface') # dim with 0 is the same so index doesn't matter
                ind_vel_34 = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,:,2,:], panel_corners_j_exp_vec[:,:,:,3,:], coll_point_i_exp_vec[:,:,:,0,:], mode='surface') # dim with 0 is the same so index doesn't matter
                ind_vel_41 = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,:,3,:], panel_corners_j_exp_vec[:,:,:,0,:], coll_point_i_exp_vec[:,:,:,0,:], mode='surface') # dim with 0 is the same so index doesn't matter

                ind_vel = ind_vel_12 + ind_vel_23 + ind_vel_34 + ind_vel_41

                ind_vel_normal = csdl.sum(ind_vel*panel_normal_i_exp_vec[:,:,:,0,:], axes=(3,))
                ind_vel_normal_mat = ind_vel_normal.reshape((num_nodes, nt, num_panels_i, num_panels_j))
                AIC_gamma = AIC_gamma.set(csdl.slice[:,:,start_i:stop_i, start_j:stop_j], value=ind_vel_normal_mat)

                start_j += num_panels_j
            start_i += num_panels_i

    # looping through the wake
    # we add part of this to the AIC_gamma 
    num_wake_panels = 0
    num_first_wake_panels = 0
    for surface in surface_names:
        num_wake_panels += wake_mesh_dict[surface]['num_panels']
        num_first_wake_panels += wake_mesh_dict[surface]['ns'] - 1

    AIC_wake = csdl.Variable(shape=(num_nodes, nt, num_tot_panels, num_wake_panels - num_first_wake_panels), value=0.)
    AIC_gamma_adjustment = csdl.Variable(shape=AIC_gamma.shape, value=0.)
    # AIC_gamma_total = csdl.Variable(shape=AIC_gamma.shape, value=0.)
    gamma = csdl.Variable(shape=(num_nodes, nt, num_tot_panels), value=0.)
    gamma_wake = csdl.Variable(shape=(num_nodes, nt, num_wake_panels), value=0.)
    gamma_wake_minus_1 = csdl.Variable(shape=(num_nodes, nt, num_wake_panels - num_first_wake_panels), value=0.)
    
    with csdl.namespace('transient solver'):
        for t in csdl.frange(nt-1):
            start_gamma, stop_gamma = 0, 0
            start_gamma_m1, stop_gamma_m1 = 0, 0

            print(f'timestep {t}')
            start_i, stop_i = 0, 0
            for i in range(num_surfaces): # surface where BC is applied

                surf_i_name = surface_names[i]
                coll_point_i = mesh_dict[surf_i_name]['panel_center'][:,t,:,:,:] # evaluation point
                panel_normal_i = mesh_dict[surf_i_name]['panel_normal'][:,t,:,:,:] # normal of collocation point
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
                    panel_normal_j = wake_mesh_dict[surf_j_name]['panel_normal'][:,t,:,:,:]

                    num_interactions = num_panels_i*num_panels_j

                    # ============ EXPANDING THE TERMS FOR SURFACE i ============
                    coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nc_i-1, ns_i-1, num_panels_j, 4, 3), 'jklm->jklabm')
                    coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, num_interactions, 4, 3))

                    panel_normal_i_exp = csdl.expand(panel_normal_i, (num_nodes, nc_i-1, ns_i-1, num_panels_j, 4, 3), 'jklm->jklabm')
                    panel_normal_i_exp_vec = panel_normal_i_exp.reshape((num_nodes, num_interactions, 4, 3))

                    # ============ EXPANDING THE TERMS FOR SURFACE j ============
                    panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklmn->jaklmn')
                    panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, num_interactions, 4, 3))

                    # panel_normal_j_exp = csdl.expand(panel_normal_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
                    # panel_normal_j_exp_vec = panel_normal_j_exp.reshape((num_nodes, num_interactions, 4, 3))

                    ind_vel_w_12 = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,0,:], panel_corners_j_exp_vec[:,:,1,:], coll_point_i_exp_vec[:,:,0,:], mode='wake')
                    ind_vel_w_23 = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,1,:], panel_corners_j_exp_vec[:,:,2,:], coll_point_i_exp_vec[:,:,0,:], mode='wake')
                    ind_vel_w_34 = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,2,:], panel_corners_j_exp_vec[:,:,3,:], coll_point_i_exp_vec[:,:,0,:], mode='wake')
                    ind_vel_w_41 = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,3,:], panel_corners_j_exp_vec[:,:,0,:], coll_point_i_exp_vec[:,:,0,:], mode='wake')

                    # NOTE: PRINT panel_corners_j_exp_vec.value[0,:10,:,:] to see how there is a panel connected from the LAST WAKE LINE to (0,0,0), creating interactions
                    # this shows up in the wake interactions term, not the kutta condition (likely causing asymmetry)
 
                    ind_vel_w = ind_vel_w_12 + ind_vel_w_23 + ind_vel_w_34 + ind_vel_w_41

                    ind_vel_w_normal = csdl.sum(ind_vel_w*panel_normal_i_exp_vec[:,:,0,:], axes=(2,))
                    # ind_vel_w_normal_mat = ind_vel_w_normal.reshape((num_nodes, nt, num_panels_i, num_panels_j))
                    ind_vel_w_normal_mat = ind_vel_w_normal.reshape((num_nodes, num_panels_i, num_panels_j))
                    AIC_wake = AIC_wake.set(csdl.slice[:,t,start_i:stop_i, start_j:stop_j], value=ind_vel_w_normal_mat[:,:,(ns_j-1):])

                    # ADJUST AIC_mu HERE 
                    kutta_condition = ind_vel_w_normal_mat[:,:,:(ns_j-1)] # gets added to AIC_mu
                    # remainder of "ind_vel_w_normal_mat" is used in the AIC_wake matrix

                    nc_j_surf, ns_j_surf = mesh_dict[surf_j_name]['nc'], mesh_dict[surf_j_name]['ns']
                    num_panels_j_surf = mesh_dict[surf_j_name]['num_panels']
                    stop_j_surf += num_panels_j_surf

                    # AIC_gamma_adjustment = AIC_gamma_adjustment.set(csdl.slice[:,t,start_i:stop_i, start_j_surf:start_j_surf+(ns_j-1)], value=-kutta_condition) # lower TE adjustment
                    AIC_gamma_adjustment = AIC_gamma_adjustment.set(csdl.slice[:,t,start_i:stop_i, (stop_j_surf-(ns_j-1)):stop_j_surf], value=kutta_condition) # upper TE adjustment
                    
                    AIC_gamma = AIC_gamma.set(csdl.slice[:,t,start_i:stop_i, start_j_surf:start_j_surf+(ns_j-1)], value=-kutta_condition+AIC_gamma[:,t,start_i:stop_i, start_j_surf:start_j_surf+(ns_j-1)])
                    AIC_gamma = AIC_gamma.set(csdl.slice[:,t,start_i:stop_i, (stop_j_surf-(ns_j-1)):stop_j_surf], value=kutta_condition+AIC_gamma[:,t,start_i:stop_i, (stop_j_surf-(ns_j-1)):stop_j_surf])
                    1

                    start_j += num_panels_j - ns_j
                    start_j_surf += num_panels_j_surf

                start_i += num_panels_i

                # ADJUSTING MU FOR THE LINEAR SYSTEM SOLVE (sub variable with the first row of wake panels missing)
                num_surf_wake_panels = wake_mesh_dict[surf_i_name]['num_panels']
                num_surf_wake_first_panels = wake_mesh_dict[surf_i_name]['ns'] - 1

                stop_gamma += num_surf_wake_panels
                stop_gamma_m1 += num_surf_wake_panels - num_surf_wake_first_panels

                gamma_wake_minus_1 = gamma_wake_minus_1.set(csdl.slice[:,t,start_gamma_m1:stop_gamma_m1], value=gamma_wake[:,t,(start_gamma+num_surf_wake_first_panels):stop_gamma])
                
                start_gamma += num_surf_wake_panels
                start_gamma_m1 += num_surf_wake_panels - num_surf_wake_first_panels
            
            # solving linear system:
            # # RHS
            # wake_influence = csdl.einsum(AIC_wake[:,t,:,:], gamma_wake[:,t,:], action='ijk,ik->ij')
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
                wake_influence = csdl.matvec(AIC_wake[nn,t,:,:], gamma_wake_minus_1[nn,t,:])
                RHS = -no_penetration_cond[nn,t,:] - wake_influence
                # RHS = -no_penetration_cond[nn,t,:]

                # AIC_gamma_total = AIC_gamma_total.set(csdl.slice[:,t,:,:], value=AIC_gamma[:,t,:,:]+AIC_gamma_adjustment[:,t,:,:])
                # gamma_timestep = csdl.solve_linear(AIC_gamma_total[nn, t, :, :], RHS)
                gamma_timestep = csdl.solve_linear(AIC_gamma[nn, t, :, :], RHS)
                gamma = gamma.set(csdl.slice[nn,t,:], value=gamma_timestep)

            if t == nt-1:
                continue
            if free_wake:
                # induced_vel = csdl.Variable(shape=(num_nodes, nt, num_wake_panels, 3)) # CHANGE IN THE FUTURE TO BE THE NODES
                induced_vel = free_wake_comp(num_nodes, t, mesh_dict, gamma, wake_mesh_dict, gamma_wake)
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
                gamma_surf_grid = gamma[:,t, start_i_s:stop_i_s].reshape((num_nodes, nc_s-1, ns_s-1))
                gamma_wake_first_row = gamma_surf_grid[:,-1,:] - gamma_surf_grid[:,0,:] # mu_upper - mu_lower @ TE

                gamma_wake_grid = gamma_wake[:,t,start_i_w:stop_i_w].reshape((num_nodes, nc_w-1, ns_w-1))
                gamma_wake_grid_next = csdl.Variable(shape=gamma_wake_grid.shape, value=0.)
                gamma_wake_grid_next = gamma_wake_grid_next.set(csdl.slice[:,0,:], value=gamma_wake_first_row)
                gamma_wake_grid_next = gamma_wake_grid_next.set(csdl.slice[:,1:,:], value=gamma_wake_grid[:,0:-1])
                # mu_wake_grid_next = mu_wake_grid_next.set(csdl.slice[:,1:,:], value=mu_wake_grid[:,0:-1])

                gamma_wake = gamma_wake.set(csdl.slice[:,t+1,start_i_w:stop_i_w], value=gamma_wake_grid_next.reshape((num_nodes, (nc_w-1)*(ns_w-1))))
                # gamma_wake = gamma_wake.set(csdl.slice[:,t+1,start_i_w+(ns_w-1):start_i_w+(ns_w-1)*(t+1)], value=gamma_wake[:,t,start_i_w:start_i_w+(ns_w-1)*t])

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

        
    # import pickle
    # filehandler = open('pm_AIC', 'wb')
    # pickle.dump(AIC_gamma.value[0,:,:,:], filehandler)
    # filehandler.close()
    # filehandler = open('pm_AIC_wake', 'wb')
    # pickle.dump(AIC_wake.value[0,:,:,:], filehandler)
    # filehandler.close()
    # filehandler = open('pm_kutta_condition', 'wb')
    # pickle.dump(kutta_condition.value[0,:,:], filehandler)
    # filehandler.close()
    # filehandler = open('pm_gamma', 'wb')
    # pickle.dump(gamma.value[0,:,:], filehandler)
    # filehandler.close()
    # print('exiting')
    # exit()

    return gamma, gamma_wake