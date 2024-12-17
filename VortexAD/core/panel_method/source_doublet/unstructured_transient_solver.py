import csdl_alpha as csdl 
import numpy as np

# from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_strengths, compute_source_influence
# from VortexAD.core.panel_method.source_doublet.doublet_functions import compute_doublet_influence, compute_doublet_influence_H_S
from VortexAD.core.panel_method.source_doublet.wake_geometry_new import wake_geometry

from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_strengths, compute_source_influence_new 
from VortexAD.core.panel_method.source_doublet.doublet_functions import compute_doublet_influence_new

from VortexAD.core.panel_method.source_doublet.free_wake_comp import free_wake_comp

def unstructured_transient_solver(mesh_dict, wake_mesh_dict, num_nodes, nt, num_tot_panels, dt, free_wake=False):
    sigma = compute_source_strengths(mesh_dict, num_nodes, nt, num_tot_panels, mesh_mode='unstructured')
    AIC_mu, AIC_sigma = static_AIC_computation(mesh_dict, num_nodes, nt, num_tot_panels)

    asdf = list(np.arange(AIC_mu.shape[2]))
    # AIC_mu = AIC_mu.set(csdl.slice[:,:,asdf,asdf], value=-0.5)
    print(AIC_mu[0,0,asdf,asdf].value)
    print(AIC_mu[0,0,0,:].value)

    sigma_BC_influence = csdl.einsum(AIC_sigma, sigma, action='ijlk,ijk->ijl')

    upper_TE_cell_ind = mesh_dict['upper_TE_cells']
    lower_TE_cell_ind = mesh_dict['lower_TE_cells']
    num_first_wake_panels = len(upper_TE_cell_ind)
    num_wake_panels = wake_mesh_dict['num_panels']

    print(sigma[0,0,upper_TE_cell_ind].value)
    print(sigma[0,0,lower_TE_cell_ind].value)
    print(AIC_sigma[0,0,upper_TE_cell_ind, upper_TE_cell_ind].value)
    print(AIC_sigma[0,0,lower_TE_cell_ind, lower_TE_cell_ind].value)
    # exit()

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
        coll_point = mesh_dict['panel_center_mod'][:,t,:,:] # (nn, num_tot_panels, 3)

        # setting up values for the wake
        # NOTE: wake mesh uses structured 4-sided panels
        panel_corners_w = wake_mesh_dict['panel_corners'][:,t,:,:,:,:] # (nn, nc_w, ns_w, 4, 3)
        coll_point_w = wake_mesh_dict['panel_center'][:,t,:,:,:] # (nn, nc_w, ns_w, 4, 3)
        panel_x_dir_w = wake_mesh_dict['panel_x_dir'][:,t,:,:,:] # (nn, nc_w, ns_w, 3)
        panel_y_dir_w = wake_mesh_dict['panel_y_dir'][:,t,:,:,:] # (nn, nc_w, ns_w, 3)
        panel_normal_w = wake_mesh_dict['panel_normal'][:,t,:,:,:] # (nn, nc_w, ns_w, 3)
        SL_w = wake_mesh_dict['SL'][:,t,:,:,:]
        SM_w = wake_mesh_dict['SM'][:,t,:,:,:]

        nc_w, ns_w = panel_corners_w.shape[1], panel_corners_w.shape[2]

        # target expansion and vectorization shapes
        num_wake_interactions = num_tot_panels*num_wake_panels
        expanded_shape = (num_nodes, num_tot_panels, nc_w, ns_w, 4, 3)
        vectorized_shape = (num_nodes, num_wake_interactions, 4, 3)

        # expanding and vectorizing terms
        coll_point_exp = csdl.expand(coll_point, (num_nodes, num_tot_panels, num_wake_panels, 4, 3), 'ijk->ijabk')
        coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

        coll_point_w_exp = csdl.expand(coll_point_w, expanded_shape, 'ijkl->iajkbl')
        coll_point_w_exp_vec = coll_point_w_exp.reshape(vectorized_shape)

        panel_corners_w_exp = csdl.expand(panel_corners_w, expanded_shape, 'ijklm->iajklm')
        panel_corners_w_exp_vec = panel_corners_w_exp.reshape(vectorized_shape)

        panel_x_dir_w_exp = csdl.expand(panel_x_dir_w, expanded_shape, 'ijkl->iajkbl')
        panel_x_dir_w_exp_vec = panel_x_dir_w_exp.reshape(vectorized_shape)
        panel_y_dir_w_exp = csdl.expand(panel_y_dir_w, expanded_shape, 'ijkl->iajkbl')
        panel_y_dir_w_exp_vec = panel_y_dir_w_exp.reshape(vectorized_shape)
        panel_normal_w_exp = csdl.expand(panel_normal_w, expanded_shape, 'ijkl->iajkbl')
        panel_normal_w_exp_vec = panel_normal_w_exp.reshape(vectorized_shape)

        SL_w_exp = csdl.expand(SL_w, expanded_shape[:-1], 'jklm->jaklm')
        SL_w_exp_vec = SL_w_exp.reshape(vectorized_shape[:-1])

        SM_w_exp = csdl.expand(SM_w, expanded_shape[:-1], 'jklm->jaklm')
        SM_w_exp_vec = SM_w_exp.reshape(vectorized_shape[:-1])

        a = coll_point_exp_vec - panel_corners_w_exp_vec # Rc - Ri
        P_JK = coll_point_exp_vec - coll_point_w_exp_vec # RcJ - RcK
        sum_ind = len(a.shape) - 1

        A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
        AL = csdl.sum(a*panel_x_dir_w_exp_vec, axes=(sum_ind,))
        AM = csdl.sum(a*panel_y_dir_w_exp_vec, axes=(sum_ind,)) # m-direction projection 
        PN = csdl.sum(P_JK*panel_normal_w_exp_vec, axes=(sum_ind,)) # normal projection of CP

        B = csdl.Variable(shape=A.shape, value=0.)
        B = B.set(csdl.slice[:,:,:-1], value=A[:,:,1:])
        B = B.set(csdl.slice[:,:,-1], value=A[:,:,0])

        BL = csdl.Variable(shape=AL.shape, value=0.)
        BL = BL.set(csdl.slice[:,:,:-1], value=BL[:,:,1:])
        BL = BL.set(csdl.slice[:,:,-1], value=BL[:,:,0])

        BM = csdl.Variable(shape=AM.shape, value=0.)
        BM = BM.set(csdl.slice[:,:,:-1], value=AM[:,:,1:])
        BM = BM.set(csdl.slice[:,:,-1], value=AM[:,:,0])

        A1 = AM*SL_w_exp_vec - AL*SM_w_exp_vec

        # print(A.shape)

        A_list = [A[:,:,ind] for ind in range(4)]
        AM_list = [AM[:,:,ind] for ind in range(4)]
        B_list = [B[:,:,ind] for ind in range(4)]
        BM_list = [BM[:,:,ind] for ind in range(4)]
        SL_list = [SL_w_exp_vec[:,:,ind] for ind in range(4)]
        SM_list = [SM_w_exp_vec[:,:,ind] for ind in range(4)]
        A1_list = [A1[:,:,ind] for ind in range(4)]
        PN_list = [PN[:,:,ind] for ind in range(4)]

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
        wake_doublet_influence = wake_doublet_influence_vec.reshape((num_nodes, num_tot_panels, num_wake_panels))
        AIC_wake = AIC_wake.set(csdl.slice[:,t,:,:], value=wake_doublet_influence[:,:,num_first_wake_panels:]) # NOTE THIS ONLY WORKS FOR SINGLE SURFACE AT THIS POINT

        kutta_condition = wake_doublet_influence[:,:,:num_first_wake_panels]

        # kc_reshaped = kutta_condition.reshape((len(lower_TE_cell_ind), num_nodes, num_tot_panels))
        kc_reshaped = kutta_condition.reshape((num_nodes, num_tot_panels, len(lower_TE_cell_ind)))
        # AIC_mu_adjustment = AIC_mu_adjustment.set(csdl.slice[:,t,:,list(lower_TE_cell_ind)], value=-kc_reshaped)
        # AIC_mu_adjustment = AIC_mu_adjustment.set(csdl.slice[:,t,:,list(upper_TE_cell_ind)], value=kc_reshaped)
        for te_ind in range(len(list(lower_TE_cell_ind))):
            AIC_mu_adjustment = AIC_mu_adjustment.set(csdl.slice[:,t,:,lower_TE_cell_ind[te_ind]], value=-kc_reshaped[:,:,te_ind])
            AIC_mu_adjustment = AIC_mu_adjustment.set(csdl.slice[:,t,:,upper_TE_cell_ind[te_ind]], value=kc_reshaped[:,:,te_ind])

        mu_wake_minus_1 = mu_wake_minus_1.set(csdl.slice[:,t,:], value=mu_wake[:,t,num_first_wake_panels:])

        # solving linear system
        AIC_mu_total = AIC_mu_total.set(csdl.slice[:,t,:,:], value=AIC_mu[:,t,:,:]+AIC_mu_adjustment[:,t,:,:])
        # AIC_mu_total = AIC_mu_total.set(csdl.slice[:,t,:,:], value=AIC_mu[:,t,:,:])
        for nn in csdl.frange(num_nodes):
            wake_influence = csdl.matvec(AIC_wake[nn,t,:,:], mu_wake_minus_1[nn,t,:])
            RHS = -sigma_BC_influence[nn,t,:] - wake_influence
            # RHS = -sigma_BC_influence[nn,t,:]
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

    coll_point_eval = mesh_dict['panel_center_mod']

    coll_point = mesh_dict['panel_center'] # (nn, nt, num_tot_panels, 3)
    panel_corners = mesh_dict['panel_corners'] # (nn, nt, num_tot_panels, 3, 3) 
    panel_x_dir = mesh_dict['panel_x_dir'] # (nn, nt, num_tot_panels, 3)
    panel_y_dir = mesh_dict['panel_y_dir'] # (nn, nt, num_tot_panels, 3)
    panel_normal = mesh_dict['panel_normal'] # (nn, nt, num_tot_panels, 3)
    S_j = mesh_dict['S']
    SL_j = mesh_dict['SL']
    SM_j = mesh_dict['SM']

    num_interactions = num_tot_panels**2
    expanded_shape = (num_nodes, nt, num_tot_panels, num_tot_panels, 3, 3)
    vectorized_shape = (num_nodes, nt, num_interactions, 3, 3)

    # expanding collocation points (where boundary condition is applied, the "i-th" expansion-vectorization)
    coll_point_exp = csdl.expand(coll_point_eval, expanded_shape, 'ijkl->ijkabl')
    coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

    # expanding the panel terms used to compute influences AT the collocation points
    # -> the "j-th" expansion-vectorization
    coll_point_j_exp = csdl.expand(coll_point, expanded_shape, 'ijkl->ijakbl')
    coll_point_j_exp_vec = coll_point_j_exp.reshape(vectorized_shape)

    panel_corners_exp = csdl.expand(panel_corners, expanded_shape, 'ijklm->ijaklm')
    panel_corners_exp_vec = panel_corners_exp.reshape(vectorized_shape)

    panel_x_dir_exp = csdl.expand(panel_x_dir, expanded_shape, 'ijkl->ijakbl')
    panel_x_dir_exp_vec = panel_x_dir_exp.reshape(vectorized_shape)
    panel_y_dir_exp = csdl.expand(panel_y_dir, expanded_shape, 'ijkl->ijakbl')
    panel_y_dir_exp_vec = panel_y_dir_exp.reshape(vectorized_shape)
    panel_normal_exp = csdl.expand(panel_normal, expanded_shape, 'ijkl->ijakbl')
    panel_normal_exp_vec = panel_normal_exp.reshape(vectorized_shape)

    # dpij_exp = csdl.expand(S_j, expanded_shape[:-1] + (2,), 'ijklm->ijaklm')
    # dpij_exp_vec = dpij_exp.reshape(vectorized_shape[:-1] + (2,))
    # dij_exp = csdl.expand(dij, expanded_shape[:-1], 'ijkl->ijakl')
    # dij_exp_vec = dij_exp.reshape(vectorized_shape[:-1])

    S_j_exp = csdl.expand(S_j, expanded_shape[:-1] , 'ijkl->ijakl')
    S_j_exp_vec = S_j_exp.reshape(vectorized_shape[:-1])

    SL_j_exp = csdl.expand(SL_j, expanded_shape[:-1], 'ijkl->ijakl')
    SL_j_exp_vec = SL_j_exp.reshape(vectorized_shape[:-1])

    SM_j_exp = csdl.expand(SM_j, expanded_shape[:-1], 'ijkl->ijakl')
    SM_j_exp_vec = SM_j_exp.reshape(vectorized_shape[:-1])

    a = coll_point_exp_vec - panel_corners_exp_vec # Rc - Ri
    P_JK = coll_point_exp_vec - coll_point_j_exp_vec # RcJ - RcK
    sum_ind = len(a.shape) - 1

    A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
    AL = csdl.sum(a*panel_x_dir_exp_vec, axes=(sum_ind,))
    AM = csdl.sum(a*panel_y_dir_exp_vec, axes=(sum_ind,)) # m-direction projection 
    PN = csdl.sum(P_JK*panel_normal_exp_vec, axes=(sum_ind,)) # normal projection of CP

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

    A_list = [A[:,:,:,ind] for ind in range(3)]
    AM_list = [AM[:,:,:,ind] for ind in range(3)]
    B_list = [B[:,:,:,ind] for ind in range(3)]
    BM_list = [BM[:,:,:,ind] for ind in range(3)]
    SL_list = [SL_j_exp_vec[:,:,:,ind] for ind in range(3)]
    SM_list = [SM_j_exp_vec[:,:,:,ind] for ind in range(3)]
    A1_list = [A1[:,:,:,ind] for ind in range(3)]
    PN_list = [PN[:,:,:,ind] for ind in range(3)]
    S_list = [S_j_exp_vec[:,:,:,ind] for ind in range(3)]

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
    doublet_influence = doublet_influence_vec.reshape((num_nodes, nt, num_tot_panels, num_tot_panels))
    AIC_mu = doublet_influence

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
    source_influence = source_influence_vec.reshape((num_nodes, nt, num_tot_panels, num_tot_panels))
    AIC_sigma = source_influence

    # source_influence_vec = compute_source_influence(
    #     dij_list,
    #     0, # we don't use this input for now
    #     dpij_list,
    #     dx_list,
    #     dy_list,
    #     dz_list,
    #     rk_list,
    #     ek_list,
    #     hk_list,
    #     mode='potential'
    # )
    # source_influence = source_influence_vec.reshape((num_nodes, nt, num_tot_panels, num_tot_panels))

    # doublet_influence_vec = compute_doublet_influence_H_S(
    #     dij_list,
    #     dpij_list,
    #     rk_list,
    #     dx_list,
    #     dy_list,
    #     dz_list
    # )
    # diag_terms = np.zeros((num_nodes, nt, num_tot_panels, num_tot_panels))
    # asdf = list(np.arange(num_tot_panels))
    # diag_terms[:,:,asdf,asdf] = 0.5
    # diag_terms = csdl.Variable(value=diag_terms)
    # doublet_influence = doublet_influence_vec.reshape((num_nodes, nt, num_tot_panels, num_tot_panels)) + diag_terms

    return AIC_mu, AIC_sigma