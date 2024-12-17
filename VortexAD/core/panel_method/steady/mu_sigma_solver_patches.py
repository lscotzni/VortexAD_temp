import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.panel_method.steady.fixed_wake_representation_patches import fixed_wake_representation_patches
from VortexAD.core.panel_method.steady.compute_source_strength_patches import compute_source_strength_patches

from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_influence_new 
from VortexAD.core.panel_method.source_doublet.doublet_functions import compute_doublet_influence_new


def mu_sigma_solver_patches(num_nodes, mesh_dict):

    surface_names = list(mesh_dict.keys())
    num_tot_panels = 0
    for surface in surface_names:
        for patch in mesh_dict[surface].keys():
            num_tot_panels += mesh_dict[surface][patch]['num_panels']

    # wake_mesh_dict = fixed_wake_representation(mesh_dict, num_nodes, wake_propagation_dt=0.001)
    wake_mesh_dict = fixed_wake_representation_patches(mesh_dict, num_nodes, wake_propagation_dt=10)

    sigma = compute_source_strength_patches(mesh_dict, num_nodes, num_panels=num_tot_panels)

    # static AIC matrices for linear system solve
    AIC_mu, AIC_sigma = AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels, surface_names)
    asdf = list(np.arange(0,AIC_mu.shape[-1]))
    # AIC_mu = AIC_mu.set(csdl.slice[0,asdf,asdf], value=0.5)
    print(AIC_mu[0,asdf,asdf].value)
    print(AIC_mu[0,0,:].value)
    # exit()

    sigma_BC_influence = csdl.einsum(AIC_sigma, sigma, action='ijk,ik->ij')

    mu = csdl.Variable(value=np.zeros(sigma.shape))
    for nn in csdl.frange(num_nodes):
        RHS = -sigma_BC_influence[nn,:]
        mu_nn = csdl.solve_linear(AIC_mu[nn,:,:], RHS)
        mu = mu.set(csdl.slice[nn,:], value=mu_nn)

    return mu, sigma, wake_mesh_dict

def AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels, surface_names):
    AIC_sigma = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
    AIC_mu = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
    num_surfaces = len(surface_names)
    start_i, stop_i = 0, 0
    for i in range(num_surfaces):
        surf_i_name = surface_names[i]
        for i_patch_name in mesh_dict[surf_i_name].keys():
            coll_point_i = mesh_dict[surf_i_name][i_patch_name]['panel_center_mod'] # evaluation point
            nc_i, ns_i = mesh_dict[surf_i_name][i_patch_name]['nc'], mesh_dict[surf_i_name][i_patch_name]['ns']
            num_panels_i = mesh_dict[surf_i_name][i_patch_name]['num_panels']
            stop_i += num_panels_i

            start_j, stop_j = 0, 0
            start_w_j, stop_w_j = 0, 0
            for j in range(num_surfaces):
                surf_j_name = surface_names[j]
                for j_patch_name in mesh_dict[surf_j_name].keys():
                    nc_j, ns_j = mesh_dict[surf_j_name][j_patch_name]['nc'], mesh_dict[surf_j_name][j_patch_name]['ns']
                    num_panels_j = mesh_dict[surf_j_name][j_patch_name]['num_panels']
                    stop_j += num_panels_j

                    panel_corners_j = mesh_dict[surf_j_name][j_patch_name]['panel_corners']
                    coll_point_j = mesh_dict[surf_j_name][j_patch_name]['panel_center']
                    panel_x_dir_j = mesh_dict[surf_j_name][j_patch_name]['panel_x_dir']
                    panel_y_dir_j = mesh_dict[surf_j_name][j_patch_name]['panel_y_dir']
                    panel_normal_j = mesh_dict[surf_j_name][j_patch_name]['panel_normal']

                    S_j = mesh_dict[surf_j_name][j_patch_name]['S']
                    SL_j = mesh_dict[surf_j_name][j_patch_name]['SL']
                    SM_j = mesh_dict[surf_j_name][j_patch_name]['SM']

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

                patch_names = list(mesh_dict[surf_j_name].keys())
                patch_type = [mesh_dict[surf_j_name][patch_name]['patch'] for patch_name in patch_names]

                if len(patch_type) == 2 and patch_type[0] is not None:
                    surface_patch_mode = 'separated'
                elif len(patch_type) == 1 and patch_type[0] == 'wrap':
                    surface_patch_mode = 'wrap'
                else:
                    continue # patches are not wake-shedding
                print(surface_patch_mode)

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
                if surface_patch_mode == 'wrap':
                    asdf = stop_j-num_panels_j
                    AIC_mu = AIC_mu.set(
                        csdl.slice[:,start_i:stop_i, asdf:(asdf+(ns_w_j-1))],
                        value=AIC_mu[:,start_i:stop_i, asdf:(asdf+(ns_w_j-1))] - doublet_influence_w
                    )

                    AIC_mu = AIC_mu.set(
                        csdl.slice[:,start_i:stop_i, (stop_j-(ns_w_j-1)):stop_j],
                        value=AIC_mu[:,start_i:stop_i, (stop_j-(ns_w_j-1)):stop_j] + doublet_influence_w
                    )
                elif surface_patch_mode == 'separated':
                    asdf = stop_j - num_panels_j
                    AIC_mu = AIC_mu.set(
                        csdl.slice[:,start_i:stop_i, (asdf-(ns_w_j-1)):asdf],
                        value=AIC_mu[:,start_i:stop_i, (asdf-(ns_w_j-1)):asdf] + doublet_influence_w
                    )

                    AIC_mu = AIC_mu.set(
                        csdl.slice[:,start_i:stop_i, (stop_j-(ns_w_j-1)):stop_j],
                        value=AIC_mu[:,start_i:stop_i, (stop_j-(ns_w_j-1)):stop_j] - doublet_influence_w[:,:,::-1]
                    )
                    

                # start_j += num_panels_j
                start_w_j += num_panels_w_j
            start_i += num_panels_i

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