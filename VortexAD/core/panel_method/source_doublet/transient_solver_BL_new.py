import numpy as np
import csdl_alpha as csdl

from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_strengths, compute_source_influence_new 
from VortexAD.core.panel_method.source_doublet.doublet_functions import compute_doublet_influence_new
# from VortexAD.core.panel_method.source_doublet.wake_geometry import wake_geometry
from VortexAD.core.panel_method.source_doublet.wake_geometry_new import wake_geometry

from VortexAD.core.panel_method.perturbation_velocity_comp import least_squares_velocity
from VortexAD.core.boundary_layer.boundary_layer_coupling_model import boundary_layer_coupling_model

from VortexAD.core.panel_method.source_doublet.free_wake_comp import free_wake_comp

def transient_solver_BL_new(mesh_dict, wake_mesh_dict, num_nodes, nt, num_tot_panels, dt, boundary_layer_coupling, free_wake=False):
    '''
    we solve for all AIC matrices here and sigma influence
    '''
    surface_names = list(mesh_dict.keys())
    num_surfaces = len(surface_names)

    sigma = compute_source_strengths(mesh_dict, num_nodes, nt, num_tot_panels, mesh_mode='structured') # shape=(num_nodes, nt, num_surf_panels)
    AIC_mu, AIC_sigma = static_AIC_computation(mesh_dict, num_nodes, nt, num_tot_panels, surface_names)
    asdf = list(np.arange(0,AIC_mu.shape[-1]))
    # AIC_mu = (AIC_mu**2)**0.5
    # AIC_mu = AIC_mu.set(csdl.slice[:,:,asdf,asdf], value=(AIC_mu[:,:,asdf,asdf]**2)**0.5)
    print(AIC_mu[0,0,asdf,asdf].value)
    # print(AIC_mu[0,0,0,:].value)
    # print(AIC_sigma[0,0,asdf,asdf].value)
    # print(AIC_sigma[0,0,0,:].value)
    # print(AIC_sigma[0,0,:,0].value)
    # exit()

    sigma_BC_influence = csdl.einsum(AIC_sigma, sigma, action='ijlk,ijk->ijl')
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

    # wake_influence_array = csdl.Variable(shape=mu.shape, value=0.) # uncomment to debug if necessary

    # BL OUTPUTS
    BL_mesh = boundary_layer_coupling[0]
    nc_BL = BL_mesh.shape[2]
    ns_BL = BL_mesh.shape[3]-1
    delta_star = csdl.Variable(value=np.zeros((num_nodes, nt, int(nc_BL*ns_BL))))
    theta = csdl.Variable(value=np.zeros(delta_star.shape))
    H = csdl.Variable(value=np.zeros(delta_star.shape))
    Cf = csdl.Variable(value=np.zeros(delta_star.shape))
    
    for t in csdl.frange(nt-1):
        start_mu, stop_mu = 0, 0
        start_mu_m1, stop_mu_m1 = 0, 0

        print(f'timestep {t}')
        start_i, stop_i = 0, 0
        for i in range(num_surfaces): # surface where BC is applied
            # print(f'i: {i}')
            surf_i_name = surface_names[i]
            coll_point_i = mesh_dict[surf_i_name]['panel_center_mod'][:,t,:,:,:] # evaluation point
            nc_i, ns_i = mesh_dict[surf_i_name]['nc'], mesh_dict[surf_i_name]['ns']
            num_panels_i = mesh_dict[surf_i_name]['num_panels']
            stop_i += num_panels_i

            start_j, stop_j = 0, 0
            start_j_surf, stop_j_surf = 0, 0
            for j in range(num_surfaces): # looping through wakes
                # print(f'j: {j}')
                surf_j_name = surface_names[j]
                nc_j, ns_j = wake_mesh_dict[surf_j_name]['nc'], wake_mesh_dict[surf_j_name]['ns']
                num_panels_j = wake_mesh_dict[surf_j_name]['num_panels']
                stop_j += num_panels_j - (ns_j-1)

                panel_corners_j = wake_mesh_dict[surf_j_name]['panel_corners'][:,t,:,:,:,:]
                coll_point_j = wake_mesh_dict[surf_j_name]['panel_center'][:,t,:,:,:]
                panel_x_dir_j = wake_mesh_dict[surf_j_name]['panel_x_dir'][:,t,:,:,:]
                panel_y_dir_j = wake_mesh_dict[surf_j_name]['panel_y_dir'][:,t,:,:,:]
                panel_normal_j = wake_mesh_dict[surf_j_name]['panel_normal'][:,t,:,:,:]

                SL_j = wake_mesh_dict[surf_j_name]['SL'][:,t,:,:,:]
                SM_j = wake_mesh_dict[surf_j_name]['SM'][:,t,:,:,:]

                num_interactions = num_panels_i*num_panels_j

                coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nc_i-1, ns_i-1, num_panels_j, 4, 3), 'jklm->jklabm')
                coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, num_interactions, 4, 3))

                panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklmn->jaklmn')
                panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, num_interactions, 4, 3))

                coll_point_j_exp = csdl.expand(coll_point_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
                coll_point_j_exp_vec = coll_point_j_exp.reshape((num_nodes, num_interactions, 4, 3))

                panel_x_dir_j_exp = csdl.expand(panel_x_dir_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
                panel_x_dir_j_exp_vec = panel_x_dir_j_exp.reshape((num_nodes, num_interactions, 4, 3))
                panel_y_dir_j_exp = csdl.expand(panel_y_dir_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
                panel_y_dir_j_exp_vec = panel_y_dir_j_exp.reshape((num_nodes, num_interactions, 4, 3))
                panel_normal_j_exp = csdl.expand(panel_normal_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
                panel_normal_j_exp_vec = panel_normal_j_exp.reshape((num_nodes, num_interactions, 4, 3))

                SL_j_exp = csdl.expand(SL_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
                SL_j_exp_vec = SL_j_exp.reshape((num_nodes, num_interactions, 4))

                SM_j_exp = csdl.expand(SM_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
                SM_j_exp_vec = SM_j_exp.reshape((num_nodes, num_interactions, 4))

                a = coll_point_i_exp_vec - panel_corners_j_exp_vec # Rc - Ri
                P_JK = coll_point_i_exp_vec - coll_point_j_exp_vec # RcJ - RcK
                sum_ind = len(a.shape) - 1

                A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
                AL = csdl.sum(a*panel_x_dir_j_exp_vec, axes=(sum_ind,))
                AM = csdl.sum(a*panel_y_dir_j_exp_vec, axes=(sum_ind,)) # m-direction projection 
                PN = csdl.sum(P_JK*panel_normal_j_exp_vec, axes=(sum_ind,)) # normal projection of CP

                B = csdl.Variable(shape=A.shape, value=0.)
                B = B.set(csdl.slice[:,:,:-1], value=A[:,:,1:])
                B = B.set(csdl.slice[:,:,-1], value=A[:,:,0])

                BL = csdl.Variable(shape=AL.shape, value=0.)
                BL = BL.set(csdl.slice[:,:,:-1], value=BL[:,:,1:])
                BL = BL.set(csdl.slice[:,:,-1], value=BL[:,:,0])

                BM = csdl.Variable(shape=AM.shape, value=0.)
                BM = BM.set(csdl.slice[:,:,:-1], value=AM[:,:,1:])
                BM = BM.set(csdl.slice[:,:,-1], value=AM[:,:,0])

                A1 = AM*SL_j_exp_vec - AL*SM_j_exp_vec

                # print(A.shape)

                A_list = [A[:,:,ind] for ind in range(4)]
                AM_list = [AM[:,:,ind] for ind in range(4)]
                B_list = [B[:,:,ind] for ind in range(4)]
                BM_list = [BM[:,:,ind] for ind in range(4)]
                SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(4)]
                SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(4)]
                A1_list = [A1[:,:,ind] for ind in range(4)]
                PN_list = [PN[:,:,ind] for ind in range(4)]

                # wake_doublet_influence_vec = compute_doublet_influence(dpij_j_exp_vec, mij_j_exp_vec, ek, hk, rk, dx, dy, dz, mode='potential')
                wake_doublet_influence_vec = compute_doublet_influence_new(
                    A_list,
                    AM_list,
                    B_list,
                    BM_list,
                    SL_list,
                    SM_list,
                    A1_list,
                    PN_list,
                    mode='potential'
                )

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
                
                # AIC_mu = AIC_mu.set(csdl.slice[:,t,start_i:stop_i, start_j_surf:start_j_surf+(ns_j-1)], value=kutta_condition+AIC_mu[:,t,start_i:stop_i, start_j_surf:start_j_surf+(ns_j-1)])
                # AIC_mu = AIC_mu.set(csdl.slice[:,t,start_i:stop_i, (stop_j_surf-(ns_j-1)):stop_j_surf], value=kutta_condition+AIC_mu[:,t,start_i:stop_i, (stop_j_surf-(ns_j-1)):stop_j_surf])
                1

                start_j += num_panels_j - (ns_j-1)
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
        
        # ============ ADD STEP HERE THAT SEGMENTS THE FIRST WAKE ROW FROM OTHERS ============
        AIC_mu_total = AIC_mu_total.set(csdl.slice[:,t,:,:], value=AIC_mu[:,t,:,:]+AIC_mu_adjustment[:,t,:,:])
        sigma_influence = csdl.einsum(AIC_sigma[:,t,:,:], sigma[:,t,:], action='jlk,jk->jl') # per timestep
        for nn in csdl.frange(num_nodes):
            # RHS
            wake_influence = csdl.matvec(AIC_wake[nn,t,:,:], mu_wake_minus_1[nn,t,:])
            # wake_influence_array = wake_influence_array.set(csdl.slice[nn,t,:], value=wake_influence)
            RHS = -sigma_influence[nn,:] - wake_influence
            
            mu_timestep = csdl.solve_linear(AIC_mu_total[nn, t, :, :], RHS)
            mu = mu.set(csdl.slice[nn,t,:], value=mu_timestep)

        # ==== LOOP FOR THE BL COUPLING ====
        num_feedback = 1 # number of times 

        sigma_orig_timestep = sigma[:,t,:]
        for fb_iter in csdl.frange(num_feedback):
            '''
            Compute edge velocities
            '''
            # NOTE: CURRENTLY HARD-CODED FOR ONE SURFACE
            # for i in range(len(num_surfaces)):
                # surface_name = surface_names[0]
            surface_name = surface_names[i]
            nc, ns = mesh_dict[surface_name]['nc'], mesh_dict[surface_name]['ns']

            delta_coll_point = mesh_dict[surface_name]['delta_coll_point'][:,t,:,:,:,:]
            mu_grid = mu[:,t,:].reshape((num_nodes, 1, nc-1, ns-1))
            ql, qm = least_squares_velocity(mu_grid, delta_coll_point.reshape((num_nodes, 1) + delta_coll_point.shape[1:]))

            panel_x_dir = mesh_dict[surface_name]['panel_x_dir'][:,t,:,:,:]
            panel_y_dir = mesh_dict[surface_name]['panel_y_dir'][:,t,:,:,:]
            panel_normal = mesh_dict[surface_name]['panel_normal'][:,t,:,:,:]

            nodal_cp_velocity = mesh_dict[surface_name]['nodal_cp_velocity'][:,t,:,:,:]
            coll_vel = mesh_dict[surface_name]['coll_point_velocity']
            if coll_vel:
                total_vel = nodal_cp_velocity+coll_vel[:,t,:,:,:]
            else:
                total_vel = nodal_cp_velocity

            free_stream_l = csdl.einsum(total_vel, panel_x_dir, action='jklm,jklm->jkl')
            free_stream_m = csdl.einsum(total_vel, panel_y_dir, action='jklm,jklm->jkl')
            free_stream_n = csdl.einsum(total_vel, panel_normal, action='jklm,jklm->jkl')

            qn = -sigma[:,t,:].reshape((num_nodes, nc-1, ns-1))

            Ql = free_stream_l + ql.reshape((num_nodes, nc-1, ns-1))
            Qm = free_stream_m + qm.reshape((num_nodes, nc-1, ns-1))
            Qn = free_stream_n + qn 
            '''
            NOTE: Qn should be zero EVEN WITH TRANSPIRATION
            - these represent the SURFACE velocities between the inviscid and viscous regions
            - transpiration is accounted for by separately adding the influence via the AIC (not directly in sigma)
            - the solution to the linear system mu will automatically include transpiration effects
            '''
            Ue_all = (Ql**2 + Qm**2 + Qn**2)**0.5
            LE_ind = int((nc-1)/2)
            Ue_ss = Ue_all[:,LE_ind:,:]
            mesh = mesh_dict[surface_name]['mesh'][:,t,:,:,:]
            mesh_ss = mesh[:,LE_ind:,:,:]

            num_BL_vel_chord = int((nc+1)/2)
            # num_BL_vel_chord = nc
            BL_vel = csdl.Variable(value=np.zeros((num_nodes, num_BL_vel_chord, ns-1))) # VELOCITY NORM
            BL_vel = BL_vel.set(csdl.slice[:,1:-1,:], value=(Ue_ss[:,1:,:]+Ue_ss[:,:-1,:])/2)
            BL_vel = BL_vel.set(csdl.slice[:,0,:], value=(Ue_all[:,LE_ind-1,:]+Ue_all[:,LE_ind,:])/2)
            BL_vel = BL_vel.set(csdl.slice[:,-1,:], value=(Ue_all[:,-1,:]+Ue_all[:,0,:])/2)

            BL_mesh = boundary_layer_coupling[0][:,t,:,:,:]
            nc_BL = BL_mesh.shape[1]
            nc_one_way = int((nc+1)/2)
            BL_per_panel = int((nc_BL-1)/(nc_one_way-1))

            BL_mesh = (BL_mesh[:,:,:-1,:]+BL_mesh[:,:,1:,:])/2. # turning from ns to ns-1
            dx = csdl.norm(BL_mesh[:,1:,:,:] - BL_mesh[:,:-1,:,:], axes=(3,))

            mesh_ss_span_center = (mesh_ss[:,:,:-1,:]+mesh_ss[:,:,1:,:])/2. # turning from ns to ns-1

            BL_vel_slope_panel = (BL_vel[:,1:,:] - BL_vel[:,:-1,:])/csdl.norm(mesh_ss_span_center[:,1:,:,:] - mesh_ss_span_center[:,:-1,:,:], axes=(3,))

            Ue = csdl.Variable(value=np.zeros((num_nodes, nc_BL, ns-1))) # edge velocity norm
            Ue = Ue.set(csdl.slice[:,::BL_per_panel,:], value=BL_vel)

            for ind in range(BL_per_panel): # looping through num BL points per panel
                interp_value = BL_vel[:,:-1,:] + BL_vel_slope_panel*csdl.norm(BL_mesh[:,ind:-1:BL_per_panel,:,:]-mesh_ss_span_center[:,:-1,:,:], axes=(3,))
                Ue = Ue.set(csdl.slice[:,ind:-1:BL_per_panel,:], value=interp_value)

            '''
            Region for BL solver
            '''
            nu = 1.52e-5
            delta_star_t, theta_t, H_t, Cf_t  = boundary_layer_coupling_model(Ue, BL_mesh, dx, nu)
            num_BL_pts = int(nc_BL*(ns-1))
            delta_star = delta_star.set(csdl.slice[:,t,:], value=delta_star_t.reshape((num_nodes, num_BL_pts)))
            theta = theta.set(csdl.slice[:,t,:], value=theta_t.reshape((num_nodes, num_BL_pts)))
            H = H.set(csdl.slice[:,t,:], value=H_t.reshape((num_nodes, num_BL_pts)))
            Cf = Cf.set(csdl.slice[:,t,:], value=Cf_t.reshape((num_nodes, num_BL_pts)))

            Ue_delta = Ue*delta_star_t

            V_transpiration_BL = csdl.Variable(value=np.zeros(Ue.shape))
            V_transpiration_BL = V_transpiration_BL.set(csdl.slice[:,1:-1,:], value=(Ue_delta[:,2:,:] - Ue_delta[:,:-2,:])/(dx[:,1:,:] + dx[:,:-1,:])) # central difference
            V_transpiration_BL = V_transpiration_BL.set(csdl.slice[:,0,:], value=(-3*Ue_delta[:,0,:] + 4*Ue_delta[:,1,:] - Ue_delta[:,2,:]) / (dx[:,0,:] + dx[:,1,:])) # backward difference
            V_transpiration_BL = V_transpiration_BL.set(csdl.slice[:,-1,:], value=(Ue_delta[:,-3,:] - 4*Ue_delta[:,-2,:] + 3*Ue_delta[:,-1,:]) / (dx[:,-2,:] + dx[:,-1,:])) # forward difference

            V_transpiration = csdl.Variable(value=np.zeros((num_nodes, nc-1, ns-1)))
            asdf = int(BL_per_panel/2)
            V_transpiration = V_transpiration.set(csdl.slice[:,LE_ind:,:], value=-V_transpiration_BL[:,asdf:-1:BL_per_panel,:])
            transp_influence = csdl.einsum(AIC_sigma[:,t,:,:], V_transpiration.reshape((num_nodes, int((nc-1)*(ns-1)))), action='jlk,jk->jl')

            sigma_coupled_influence = sigma_influence + transp_influence

            for nn in csdl.frange(num_nodes):
                # RHS
                wake_influence = csdl.matvec(AIC_wake[nn,t,:,:], mu_wake_minus_1[nn,t,:])
                # wake_influence_array = wake_influence_array.set(csdl.slice[nn,t,:], value=wake_influence)
                RHS = -sigma_coupled_influence[nn,:] - wake_influence
                # RHS = -sigma_BC_influence[nn,t,:]
                
                mu_timestep = csdl.solve_linear(AIC_mu_total[nn, t, :, :], RHS)
                # mu_timestep = csdl.solve_linear(AIC_mu[nn, t, :, :], RHS)
                mu = mu.set(csdl.slice[nn,t,:], value=mu_timestep)

            1

        if t == nt-1:
            continue
        if free_wake:
            # induced_vel = csdl.Variable(shape=(num_nodes, nt, num_wake_panels, 3)) # CHANGE IN THE FUTURE TO BE THE NODES
            induced_vel = free_wake_comp(num_nodes, t, mesh_dict, mu, sigma, wake_mesh_dict, mu_wake)
            # print(induced_vel.shape)

        # update mu and position of mesh
        start_i_s, stop_i_s = 0, 0
        start_i_w, stop_i_w = 0, 0
        start_i_w_pts, stop_i_w_pts = 0, 0
        for i in range(num_surfaces):
            surface_name = surface_names[i]
            nc_s, ns_s =  mesh_dict[surface_name]['nc'], mesh_dict[surface_name]['ns']
            num_panels_s = mesh_dict[surface_name]['num_panels']
            stop_i_s += num_panels_s

            nc_w, ns_w =  wake_mesh_dict[surface_name]['nc'], wake_mesh_dict[surface_name]['ns']
            num_panels_w = wake_mesh_dict[surface_name]['num_panels']
            # stop_i_w += num_panels_w - (ns_w-1)
            stop_i_w += num_panels_w
            num_pts_w = nc_w*ns_w
            stop_i_w_pts += num_pts_w

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
                total_vel = wake_velocity_timestep - induced_vel[:,start_i_w_pts:stop_i_w_pts,:].reshape((num_nodes, nc_w, ns_w, 3))
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

            start_i_s += num_panels_s
            start_i_w += num_panels_w
            start_i_w_pts += num_pts_w
        1
    print(mu[0,0,:].value)
    # exit()
    return mu, sigma, mu_wake

def static_AIC_computation(mesh_dict, num_nodes, nt, num_tot_panels, surface_names):
    AIC_sigma = csdl.Variable(shape=(num_nodes, nt, num_tot_panels, num_tot_panels), value=0.)
    AIC_mu = csdl.Variable(shape=(num_nodes, nt, num_tot_panels, num_tot_panels), value=0.)
    num_surfaces = len(surface_names)
    start_i, stop_i = 0, 0
    for i in range(num_surfaces):
        surf_i_name = surface_names[i]
        coll_point_i = mesh_dict[surf_i_name]['panel_center_mod'] # evaluation point
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
            coll_point_j = mesh_dict[surf_j_name]['panel_center']
            panel_x_dir_j = mesh_dict[surf_j_name]['panel_x_dir']
            panel_y_dir_j = mesh_dict[surf_j_name]['panel_y_dir']
            panel_normal_j = mesh_dict[surf_j_name]['panel_normal']

            S_j = mesh_dict[surf_j_name]['S']
            SL_j = mesh_dict[surf_j_name]['SL']
            SM_j = mesh_dict[surf_j_name]['SM']

            num_interactions = num_panels_i*num_panels_j

            coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nt, nc_i-1, ns_i-1, num_panels_j, 4, 3), 'ijklm->ijklabm')
            coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, nt, num_interactions, 4, 3))

            panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklmn->ijaklmn')
            panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))

            coll_point_j_exp = csdl.expand(coll_point_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklm->ijaklbm')
            coll_point_j_exp_vec = coll_point_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))

            panel_x_dir_j_exp = csdl.expand(panel_x_dir_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklm->ijaklbm')
            panel_x_dir_j_exp_vec = panel_x_dir_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))
            panel_y_dir_j_exp = csdl.expand(panel_y_dir_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklm->ijaklbm')
            panel_y_dir_j_exp_vec = panel_y_dir_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))
            panel_normal_j_exp = csdl.expand(panel_normal_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklm->ijaklbm')
            panel_normal_j_exp_vec = panel_normal_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))

            S_j_exp = csdl.expand(S_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4), 'ijklm->ijaklm')
            S_j_exp_vec = S_j_exp.reshape((num_nodes, nt, num_interactions, 4))

            SL_j_exp = csdl.expand(SL_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4), 'ijklm->ijaklm')
            SL_j_exp_vec = SL_j_exp.reshape((num_nodes, nt, num_interactions, 4))

            SM_j_exp = csdl.expand(SM_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4), 'ijklm->ijaklm')
            SM_j_exp_vec = SM_j_exp.reshape((num_nodes, nt, num_interactions, 4))
            
            a = coll_point_i_exp_vec - panel_corners_j_exp_vec # Rc - Ri
            P_JK = coll_point_i_exp_vec - coll_point_j_exp_vec # RcJ - RcK
            sum_ind = len(a.shape) - 1

            A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
            AL = csdl.sum(a*panel_x_dir_j_exp_vec, axes=(sum_ind,))
            AM = csdl.sum(a*panel_y_dir_j_exp_vec, axes=(sum_ind,)) # m-direction projection 
            PN = csdl.sum(P_JK*panel_normal_j_exp_vec, axes=(sum_ind,)) # normal projection of CP

            B = csdl.Variable(shape=A.shape, value=0.)
            B = B.set(csdl.slice[:,:,:,:-1], value=A[:,:,:,1:])
            B = B.set(csdl.slice[:,:,:,-1], value=A[:,:,:,0])

            BL = csdl.Variable(shape=AL.shape, value=0.)
            BL = BL.set(csdl.slice[:,:,:,:-1], value=BL[:,:,:,1:])
            BL = BL.set(csdl.slice[:,:,:,-1], value=BL[:,:,:,0])

            BM = csdl.Variable(shape=AM.shape, value=0.)
            BM = BM.set(csdl.slice[:,:,:,:-1], value=AM[:,:,:,1:])
            BM = BM.set(csdl.slice[:,:,:,-1], value=AM[:,:,:,0])

            A1 = AM*SL_j_exp_vec - AL*SM_j_exp_vec

            # print(A.shape)

            A_list = [A[:,:,:,ind] for ind in range(4)]
            AM_list = [AM[:,:,:,ind] for ind in range(4)]
            B_list = [B[:,:,:,ind] for ind in range(4)]
            BM_list = [BM[:,:,:,ind] for ind in range(4)]
            SL_list = [SL_j_exp_vec[:,:,:,ind] for ind in range(4)]
            SM_list = [SM_j_exp_vec[:,:,:,ind] for ind in range(4)]
            A1_list = [A1[:,:,:,ind] for ind in range(4)]
            PN_list = [PN[:,:,:,ind] for ind in range(4)]
            S_list = [S_j_exp_vec[:,:,:,ind] for ind in range(4)]

            doublet_influence_vec = compute_doublet_influence_new(
                A_list, 
                AM_list, 
                B_list, 
                BM_list, 
                SL_list, 
                SM_list, 
                A1_list, 
                PN_list, 
                mode='potential'
            )
            doublet_influence = doublet_influence_vec.reshape((num_nodes, nt, num_panels_i, num_panels_j))
            AIC_mu = AIC_mu.set(csdl.slice[:,:,start_i:stop_i, start_j:stop_j], value=doublet_influence)

            source_influence_vec = compute_source_influence_new(
                A_list, 
                AM_list, 
                B_list, 
                BM_list, 
                SL_list, 
                SM_list, 
                A1_list, 
                PN_list, 
                S_list, 
                mode='potential'
            )
            source_influence = source_influence_vec.reshape((num_nodes, nt, num_panels_i, num_panels_j))
            AIC_sigma = AIC_sigma.set(csdl.slice[:,:,start_i:stop_i, start_j:stop_j], value=source_influence)

            start_j += num_panels_j
        start_i += num_panels_i

    return AIC_mu, AIC_sigma