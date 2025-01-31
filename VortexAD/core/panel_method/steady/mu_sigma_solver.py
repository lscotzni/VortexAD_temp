import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.panel_method.steady.fixed_wake_representation import fixed_wake_representation
from VortexAD.core.panel_method.steady.compute_source_strength import compute_source_strength

from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_influence_new 
from VortexAD.core.panel_method.source_doublet.doublet_functions import compute_doublet_influence_new


def mu_sigma_solver(num_nodes, mesh_dict, mode='structured', bc='Dirichlet'):

    if mode == 'structured':
        surface_names = list(mesh_dict.keys())
        num_tot_panels = 0
        for surface in surface_names:
            num_tot_panels += mesh_dict[surface]['num_panels']
    
    elif mode == 'unstructured':
        num_tot_panels = len(mesh_dict['cell_adjacency'])

    # wake_mesh_dict = fixed_wake_representation(mesh_dict, num_nodes, wake_propagation_dt=0.001)
    wake_mesh_dict = fixed_wake_representation(mesh_dict, num_nodes, wake_propagation_dt=10, mesh_mode=mode)

    sigma = compute_source_strength(mesh_dict, num_nodes, num_panels=num_tot_panels, mesh_mode=mode)
    # graph = csdl.get_current_recorder().active_graph
    # graph.visualize('pre-subgraph')
    # static AIC matrices for linear system solve
    if mode == 'structured':
        if bc == 'Dirichlet':
            AIC_mu, AIC_sigma = AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels, surface_names)
        elif bc == 'Neumann':
            AIC_mu, AIC_sigma = Neumann_AIC_computation()
    elif mode == 'unstructured':
        AIC_mu, AIC_sigma = unstructured_AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels)
        # for i in csdl.frange(1):
        #     AIC_mu, AIC_sigma = unstructured_AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels)
        #     graph = csdl.get_current_recorder().active_graph
        # graph.visualize('subgraph')
        # csdl.get_current_recorder().visualize_graph('entire_graph', visualize_style='hierarchical')
        # exit()

    else:
        raise ValueError('Mode must be structured or unstructured')
    asdf = list(np.arange(0,AIC_mu.shape[-1]))
    # AIC_mu = AIC_mu.set(csdl.slice[0,asdf,asdf], value=0.5)
    # print(AIC_mu[0,asdf,asdf].value)
    # print(AIC_mu[0,0,:].value)
    # exit()

    sigma_BC_influence = csdl.einsum(AIC_sigma, sigma, action='ijk,ik->ij')

    mu = csdl.Variable(value=np.zeros(sigma.shape))
    for nn in csdl.frange(num_nodes):
        RHS = -sigma_BC_influence[nn,:]
        mu_nn = csdl.solve_linear(AIC_mu[nn,:,:], RHS)
        mu = mu.set(csdl.slice[nn,:], value=mu_nn)

    return mu, sigma, wake_mesh_dict, AIC_mu, AIC_sigma

def AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels, surface_names):
    AIC_sigma = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
    AIC_mu = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
    num_surfaces = len(surface_names)
    start_i, stop_i = 0, 0
    for i in range(num_surfaces):
        surf_i_name = surface_names[i]

        coll_point_i = mesh_dict[surf_i_name]['panel_center_mod'] # evaluation point
        nc_i, ns_i = mesh_dict[surf_i_name]['nc'], mesh_dict[surf_i_name]['ns']
        num_panels_i = mesh_dict[surf_i_name]['num_panels']
        stop_i += num_panels_i

        start_j, stop_j = 0, 0
        start_w_j, stop_w_j = 0, 0
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

            S_j_exp = csdl.expand(S_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
            S_j_exp_vec = S_j_exp.reshape((num_nodes, num_interactions, 4))

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

            A_list = [A[:,:,ind] for ind in range(4)]
            AM_list = [AM[:,:,ind] for ind in range(4)]
            B_list = [B[:,:,ind] for ind in range(4)]
            BM_list = [BM[:,:,ind] for ind in range(4)]
            SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(4)]
            SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(4)]
            A1_list = [A1[:,:,ind] for ind in range(4)]
            PN_list = [PN[:,:,ind] for ind in range(4)]
            S_list = [S_j_exp_vec[:,:,ind] for ind in range(4)]

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
            doublet_influence = doublet_influence_vec.reshape((num_nodes, num_panels_i, num_panels_j))
            AIC_mu = AIC_mu.set(csdl.slice[:,start_i:stop_i, start_j:stop_j], value=doublet_influence)

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
            source_influence = source_influence_vec.reshape((num_nodes, num_panels_i, num_panels_j))
            AIC_sigma = AIC_sigma.set(csdl.slice[:,start_i:stop_i, start_j:stop_j], value=source_influence)
            start_j += num_panels_j

            # ================ wake influence here ================

            nc_w_j, ns_w_j = wake_mesh_dict[surf_j_name]['nc'], wake_mesh_dict[surf_j_name]['ns']
            num_panels_w_j = wake_mesh_dict[surf_j_name]['num_panels']
            stop_w_j += num_panels_w_j

            panel_corners_j = wake_mesh_dict[surf_j_name]['panel_corners']
            coll_point_j = wake_mesh_dict[surf_j_name]['panel_center']
            panel_x_dir_j = wake_mesh_dict[surf_j_name]['panel_x_dir']
            panel_y_dir_j = wake_mesh_dict[surf_j_name]['panel_y_dir']
            panel_normal_j = wake_mesh_dict[surf_j_name]['panel_normal']

            # S_j = wake_mesh_dict[surf_j_name]['S']
            SL_j = wake_mesh_dict[surf_j_name]['SL']
            SM_j = wake_mesh_dict[surf_j_name]['SM']

            num_interactions_w = num_panels_i*num_panels_w_j

            coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nc_i-1, ns_i-1, num_panels_w_j, 4, 3), 'jklm->jklabm')
            coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, num_interactions_w, 4, 3))

            panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4, 3), 'jklmn->jaklmn')
            panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, num_interactions_w, 4, 3))

            coll_point_j_exp = csdl.expand(coll_point_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4, 3), 'jklm->jaklbm')
            coll_point_j_exp_vec = coll_point_j_exp.reshape((num_nodes, num_interactions_w, 4, 3))

            panel_x_dir_j_exp = csdl.expand(panel_x_dir_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4, 3), 'jklm->jaklbm')
            panel_x_dir_j_exp_vec = panel_x_dir_j_exp.reshape((num_nodes, num_interactions_w, 4, 3))
            panel_y_dir_j_exp = csdl.expand(panel_y_dir_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4, 3), 'jklm->jaklbm')
            panel_y_dir_j_exp_vec = panel_y_dir_j_exp.reshape((num_nodes, num_interactions_w, 4, 3))
            panel_normal_j_exp = csdl.expand(panel_normal_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4, 3), 'jklm->jaklbm')
            panel_normal_j_exp_vec = panel_normal_j_exp.reshape((num_nodes, num_interactions_w, 4, 3))

            # S_j_exp = csdl.expand(S_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
            # S_j_exp_vec = S_j_exp.reshape((num_nodes, num_interactions, 4))

            SL_j_exp = csdl.expand(SL_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4), 'jklm->jaklm')
            SL_j_exp_vec = SL_j_exp.reshape((num_nodes, num_interactions_w, 4))

            SM_j_exp = csdl.expand(SM_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4), 'jklm->jaklm')
            SM_j_exp_vec = SM_j_exp.reshape((num_nodes, num_interactions_w, 4))
            
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

            A_list = [A[:,:,ind] for ind in range(4)]
            AM_list = [AM[:,:,ind] for ind in range(4)]
            B_list = [B[:,:,ind] for ind in range(4)]
            BM_list = [BM[:,:,ind] for ind in range(4)]
            SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(4)]
            SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(4)]
            A1_list = [A1[:,:,ind] for ind in range(4)]
            PN_list = [PN[:,:,ind] for ind in range(4)]
            # S_list = [S_j_exp_vec[:,:,ind] for ind in range(4)]

            doublet_influence_w_vec = compute_doublet_influence_new(
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
            doublet_influence_w = doublet_influence_w_vec.reshape((num_nodes, num_panels_i, num_panels_w_j))

            # doublet_influence_KC = csdl.Variable(value=np.zeros(shape=doublet_influence.shape))
            # doublet_influence_KC = doublet_influence_KC.set(
            #     csdl.slice[:,:,:],
            #     value=doublet_influence
            # )
            # doublet_influence_KC = doublet_influence_KC.set(
            #     csdl.slice[:,:,:(ns_j-1)],
            #     value=doublet_influence[:,:,:(ns_j-1)]-doublet_influence_w
            # )
            # doublet_influence_KC = doublet_influence_KC.set(
            #     csdl.slice[:,:,-(ns_j-1):],
            #     value=doublet_influence[:,:,-(ns_j-1):]+doublet_influence_w
            # )

            asdf = stop_j-num_panels_j
            AIC_mu = AIC_mu.set(
                csdl.slice[:,start_i:stop_i, asdf:(asdf+(ns_w_j-1))],
                value=AIC_mu[:,start_i:stop_i, asdf:(asdf+(ns_w_j-1))] - doublet_influence_w
            )

            AIC_mu = AIC_mu.set(
                csdl.slice[:,start_i:stop_i, (stop_j-(ns_w_j-1)):stop_j],
                value=AIC_mu[:,start_i:stop_i, (stop_j-(ns_w_j-1)):stop_j] + doublet_influence_w
            )

            # start_j += num_panels_j
            start_w_j += num_panels_w_j
        start_i += num_panels_i

    return AIC_mu, AIC_sigma

def Neumann_AIC_computation():
    #I M LUCA, I M SMART, I M THE BEST
    return AIC_mu, AIC_sigma
    
def unstructured_AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels):
    
    upper_TE_cell_ind = mesh_dict['upper_TE_cells']
    lower_TE_cell_ind = mesh_dict['lower_TE_cells']
    num_first_wake_panels = len(upper_TE_cell_ind)
    num_wake_panels = wake_mesh_dict['num_panels']

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
    expanded_shape = (num_nodes, num_tot_panels, num_tot_panels, 3, 3)
    vectorized_shape = (num_nodes, num_interactions, 3, 3)

    # expanding collocation points (where boundary condition is applied, the "i-th" expansion-vectorization)
    coll_point_exp = csdl.expand(coll_point_eval, expanded_shape, 'jkl->jkabl')
    coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

    # expanding the panel terms used to compute influences AT the collocation points
    # -> the "j-th" expansion-vectorization
    coll_point_j_exp = csdl.expand(coll_point, expanded_shape, 'jkl->jakbl')
    coll_point_j_exp_vec = coll_point_j_exp.reshape(vectorized_shape)

    panel_corners_exp = csdl.expand(panel_corners, expanded_shape, 'jklm->jaklm')
    panel_corners_exp_vec = panel_corners_exp.reshape(vectorized_shape)

    panel_x_dir_exp = csdl.expand(panel_x_dir, expanded_shape, 'jkl->jakbl')
    panel_x_dir_exp_vec = panel_x_dir_exp.reshape(vectorized_shape)
    panel_y_dir_exp = csdl.expand(panel_y_dir, expanded_shape, 'jkl->jakbl')
    panel_y_dir_exp_vec = panel_y_dir_exp.reshape(vectorized_shape)
    panel_normal_exp = csdl.expand(panel_normal, expanded_shape, 'jkl->jakbl')
    panel_normal_exp_vec = panel_normal_exp.reshape(vectorized_shape)

    # dpij_exp = csdl.expand(S_j, expanded_shape[:-1] + (2,), 'ijklm->ijaklm')
    # dpij_exp_vec = dpij_exp.reshape(vectorized_shape[:-1] + (2,))
    # dij_exp = csdl.expand(dij, expanded_shape[:-1], 'ijkl->ijakl')
    # dij_exp_vec = dij_exp.reshape(vectorized_shape[:-1])

    S_j_exp = csdl.expand(S_j, expanded_shape[:-1] , 'jkl->jakl')
    S_j_exp_vec = S_j_exp.reshape(vectorized_shape[:-1])

    SL_j_exp = csdl.expand(SL_j, expanded_shape[:-1], 'jkl->jakl')
    SL_j_exp_vec = SL_j_exp.reshape(vectorized_shape[:-1])

    SM_j_exp = csdl.expand(SM_j, expanded_shape[:-1], 'jkl->jakl')
    SM_j_exp_vec = SM_j_exp.reshape(vectorized_shape[:-1])

    a = coll_point_exp_vec - panel_corners_exp_vec # Rc - Ri
    P_JK = coll_point_exp_vec - coll_point_j_exp_vec # RcJ - RcK
    sum_ind = len(a.shape) - 1

    # for i in csdl.frange(1):

    A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
    AL = csdl.sum(a*panel_x_dir_exp_vec, axes=(sum_ind,))
    AM = csdl.sum(a*panel_y_dir_exp_vec, axes=(sum_ind,)) # m-direction projection 
    PN = csdl.sum(P_JK*panel_normal_exp_vec, axes=(sum_ind,)) # normal projection of CP

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

    A_list = [A[:,:,ind] for ind in range(3)]
    AM_list = [AM[:,:,ind] for ind in range(3)]
    B_list = [B[:,:,ind] for ind in range(3)]
    BM_list = [BM[:,:,ind] for ind in range(3)]
    SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(3)]
    SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(3)]
    A1_list = [A1[:,:,ind] for ind in range(3)]
    PN_list = [PN[:,:,ind] for ind in range(3)]
    S_list = [S_j_exp_vec[:,:,ind] for ind in range(3)]

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
    doublet_influence = doublet_influence_vec.reshape((num_nodes, num_tot_panels, num_tot_panels))
#     graph = csdl.get_current_recorder().active_graph
    # graph.visualize('subgraph_lo')
    # print(len(graph.node_table))
    # exit()
    AIC_mu_orig = doublet_influence

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
    source_influence = source_influence_vec.reshape((num_nodes, num_tot_panels, num_tot_panels))
    AIC_sigma = source_influence

    # wake AIC influence
    # location where wake influence is computed
    coll_point = mesh_dict['panel_center_mod'][:,:,:]

    # wake values
    panel_corners_w = wake_mesh_dict['panel_corners'] # (nn, nc_w, ns_w, 4, 3)
    coll_point_w = wake_mesh_dict['panel_center'] # (nn, nc_w, ns_w, 4, 3)
    panel_x_dir_w = wake_mesh_dict['panel_x_dir'] # (nn, nc_w, ns_w, 3)
    panel_y_dir_w = wake_mesh_dict['panel_y_dir'] # (nn, nc_w, ns_w, 3)
    panel_normal_w = wake_mesh_dict['panel_normal'] # (nn, nc_w, ns_w, 3)
    SL_w = wake_mesh_dict['SL']
    SM_w = wake_mesh_dict['SM']
    
    nc_w, ns_w = panel_corners_w.shape[1], panel_corners_w.shape[2]

    # target expansion and vectorization shapes
    # TODO: add support for multisurface
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
    AIC_mu_adjustment = csdl.Variable(value=np.zeros(AIC_mu_orig.shape))
    # for te_ind in range(len(list(lower_TE_cell_ind))):
    te_ind_list = list(np.arange(len(list(lower_TE_cell_ind)), dtype=int))
    print(te_ind_list)
    print(lower_TE_cell_ind)
    print(upper_TE_cell_ind)
    for te_ind, lower_ind, upper_ind in csdl.frange(vals=(te_ind_list, lower_TE_cell_ind, upper_TE_cell_ind)):
        # lower_ind = lower_TE_cell_ind[te_ind]
        # upper_ind = upper_TE_cell_ind[te_ind]
        AIC_mu_adjustment = AIC_mu_adjustment.set(
            csdl.slice[:,:,lower_ind],
            value=-wake_doublet_influence[:,:,te_ind]
        )
        AIC_mu_adjustment = AIC_mu_adjustment.set(
            csdl.slice[:,:,upper_ind],
            value=wake_doublet_influence[:,:,te_ind]
        )

    AIC_mu = AIC_mu_orig + AIC_mu_adjustment

    return AIC_mu, AIC_sigma




'''
Structure of unsteady panel solver
- initialize wake model
    - add the first set of wake panels propagated with the free-stream to about 0.2-0.3 of the distance it should travel back
- compute source strengths (& corresponding AIC)
    - doesn't need time-stepping because we have the free-stream velocity and the time-dependent mesh
- time-stepping solver
    - compute doublet AIC
    - compute kutta condition influence on AIC
    - propagate wake
        - FREE WAKE: RECOMPUTE AIC FOR WAKES AND COMPUTE INDUCED VELOCITIES
    - move to next time step

- return mu, sigma

'''