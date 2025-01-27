import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_influence_new
from VortexAD.core.panel_method.vortex_ring.vortex_line_functions import compute_vortex_line_ind_vel

def free_wake_comp(num_nodes, t, mesh_dict, mu, sigma, wake_mesh_dict, mu_wake):
    surface_names = list(mesh_dict.keys())
    num_wake_nodes = 0
    num_wake_panels = 0
    num_surf_panels = 0
    for surface in surface_names:
        nc_w, ns_w = wake_mesh_dict[surface]['mesh'].shape[-3], wake_mesh_dict[surface]['mesh'].shape[-2]
        num_wake_nodes += nc_w*ns_w
        
        num_wake_panels += wake_mesh_dict[surface]['num_panels']

        num_surf_panels += mesh_dict[surface]['num_panels']

    AIC_mu_surf = csdl.Variable(shape=(num_nodes, num_wake_nodes, num_surf_panels, 3), value=0.)
    AIC_sigma = csdl.Variable(shape=(num_nodes, num_wake_nodes, num_surf_panels, 3), value=0.)
    AIC_mu_wake = csdl.Variable(shape=(num_nodes, num_wake_nodes, num_wake_panels, 3), value=0.)

    induced_vel = csdl.Variable(shape=(num_nodes, num_wake_nodes, 3), value=0.)
    start_i, stop_i = 0, 0
    for i in range(len(surface_names)):
        surf_name_i = surface_names[i]
        eval_pt_i = wake_mesh_dict[surf_name_i]['mesh'][:,t,:,:,:] # evaluating at the mesh nodes
        # eval_pt_i = wake_mesh_dict[surf_name_i]['mesh'][:,t,2:,:,:] # evaluating at the mesh nodes (except for 0 and 1 in the chordwise direction)
        nc_i, ns_i = wake_mesh_dict[surf_name_i]['nc'], wake_mesh_dict[surf_name_i]['ns']
        num_points_i = wake_mesh_dict[surf_name_i]['num_points']
        stop_i += num_points_i

        start_s_j, stop_s_j = 0, 0
        start_w_j, stop_w_j = 0, 0
        for j in range(len(surface_names)):
            surf_name_j = surface_names[j]

            # ==================== assembling AIC for surface doublets and sources ====================
            nc_s_j, ns_s_j = mesh_dict[surf_name_j]['nc'], mesh_dict[surf_name_j]['ns']
            num_panels_s_j = mesh_dict[surf_name_j]['num_panels']
            stop_s_j += num_panels_s_j

            panel_corners_s_j = mesh_dict[surf_name_j]['panel_corners'][:,t,:,:,:,:]
            coll_point_j = mesh_dict[surf_name_j]['panel_center'][:,t,:,:,:]
            panel_x_dir_s_j = mesh_dict[surf_name_j]['panel_x_dir'][:,t,:,:,:]
            panel_y_dir_s_j = mesh_dict[surf_name_j]['panel_y_dir'][:,t,:,:,:]
            panel_normal_s_j = mesh_dict[surf_name_j]['panel_normal'][:,t,:,:,:]

            S_j = mesh_dict[surf_name_j]['S'][:,t,:,:,:]
            SL_j = mesh_dict[surf_name_j]['SL'][:,t,:,:,:]
            SM_j = mesh_dict[surf_name_j]['SM'][:,t,:,:,:]

            num_surf_interactions = num_points_i*num_panels_s_j

            eval_pt_w_i_exp = csdl.expand(eval_pt_i, (num_nodes, nc_i, ns_i, num_panels_s_j, 4, 3), 'ijkl->ijkabl')
            eval_pt_w_i_exp_vec = eval_pt_w_i_exp.reshape((num_nodes, num_surf_interactions, 4, 3))

            panel_corners_s_j_exp = csdl.expand(panel_corners_s_j, (num_nodes, num_points_i, nc_s_j-1, ns_s_j-1, 4, 3), 'ijklm->iajklm')
            panel_corners_s_j_exp_vec = panel_corners_s_j_exp.reshape((num_nodes, num_surf_interactions, 4, 3))

            coll_point_j_exp = csdl.expand(coll_point_j, (num_nodes, num_points_i, nc_s_j-1, ns_s_j-1, 4, 3), 'ijkl->iajkbl')
            coll_point_j_exp_vec = coll_point_j_exp.reshape((num_nodes, num_surf_interactions, 4, 3))

            panel_x_dir_s_j_exp = csdl.expand(panel_x_dir_s_j, (num_nodes, num_points_i, nc_s_j-1, ns_s_j-1, 4, 3), 'ijkl->iajkbl')
            panel_x_dir_s_j_exp_vec = panel_x_dir_s_j_exp.reshape((num_nodes, num_surf_interactions, 4, 3))

            panel_y_dir_s_j_exp = csdl.expand(panel_y_dir_s_j, (num_nodes, num_points_i, nc_s_j-1, ns_s_j-1, 4, 3), 'ijkl->iajkbl')
            panel_y_dir_s_j_exp_vec = panel_y_dir_s_j_exp.reshape((num_nodes, num_surf_interactions, 4, 3))

            panel_normal_s_j_exp = csdl.expand(panel_normal_s_j, (num_nodes, num_points_i, nc_s_j-1, ns_s_j-1, 4, 3), 'ijkl->iajkbl')
            panel_normal_s_j_exp_vec = panel_normal_s_j_exp.reshape((num_nodes, num_surf_interactions, 4, 3))

            S_j_exp = csdl.expand(S_j, (num_nodes, num_points_i, nc_s_j-1, ns_s_j-1, 4), 'ijkl->iajkl')
            S_j_exp_vec = S_j_exp.reshape((num_nodes, num_surf_interactions, 4))

            SL_j_exp = csdl.expand(SL_j, (num_nodes, num_points_i, nc_s_j-1, ns_s_j-1, 4), 'ijkl->iajkl')
            SL_j_exp_vec = SL_j_exp.reshape((num_nodes, num_surf_interactions, 4))

            SM_j_exp = csdl.expand(SM_j, (num_nodes, num_points_i, nc_s_j-1, ns_s_j-1, 4), 'ijkl->iajkl')
            SM_j_exp_vec = SM_j_exp.reshape((num_nodes, num_surf_interactions, 4))

            a = eval_pt_w_i_exp_vec - panel_corners_s_j_exp_vec # Rc - Ri
            P_JK = eval_pt_w_i_exp_vec - coll_point_j_exp_vec # RcJ - RcK
            sum_ind = len(a.shape) - 1

            A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
            AL = csdl.sum(a*panel_x_dir_s_j_exp_vec, axes=(sum_ind,))
            AM = csdl.sum(a*panel_y_dir_s_j_exp_vec, axes=(sum_ind,)) # m-direction projection 
            PN = csdl.sum(P_JK*panel_normal_s_j_exp_vec, axes=(sum_ind,)) # normal projection of CP

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
            A = A.expand(panel_normal_s_j_exp_vec.shape, 'ijk->ijka')
            AM = AM.expand(panel_normal_s_j_exp_vec.shape, 'ijk->ijka')
            B = B.expand(panel_normal_s_j_exp_vec.shape, 'ijk->ijka')
            BM = BM.expand(panel_normal_s_j_exp_vec.shape, 'ijk->ijka')
            SL_j_exp_vec = SL_j_exp_vec.expand(panel_normal_s_j_exp_vec.shape, 'ijk->ijka')
            SM_j_exp_vec = SM_j_exp_vec.expand(panel_normal_s_j_exp_vec.shape, 'ijk->ijka')
            A1 = A1.expand(panel_normal_s_j_exp_vec.shape, 'ijk->ijka')
            PN = PN.expand(panel_normal_s_j_exp_vec.shape, 'ijk->ijka')
            S_j_exp_vec = S_j_exp_vec.expand(panel_normal_s_j_exp_vec.shape, 'ijk->ijka')

            A_list = [A[:,:,ind] for ind in range(4)]
            AM_list = [AM[:,:,ind] for ind in range(4)]
            B_list = [B[:,:,ind] for ind in range(4)]
            BM_list = [BM[:,:,ind] for ind in range(4)]
            SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(4)]
            SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(4)]
            A1_list = [A1[:,:,ind] for ind in range(4)]
            PN_list = [PN[:,:,ind] for ind in range(4)]
            S_list = [S_j_exp_vec[:,:,ind] for ind in range(4)]

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
                panel_x_dir_s_j_exp_vec[:,:,0,:], # these don't change along dimension 2
                panel_y_dir_s_j_exp_vec[:,:,0,:],
                panel_normal_s_j_exp_vec[:,:,0,:],
                mode='velocity'
            )

            rot_mat_s_j = mesh_dict[surf_name_j]['rot_mat'][:,t,:,:,:,:]
            rot_mat_s_j_exp = csdl.expand(rot_mat_s_j, (num_nodes, num_points_i, nc_s_j-1, ns_s_j-1, 3, 3), 'ijklm->iajkml')
            rot_mat_s_j_exp_vec = rot_mat_s_j_exp.reshape(((num_nodes, num_surf_interactions, 3, 3)))

            # ind_vel_source_global = csdl.einsum(ind_vel_source_local, rot_mat_s_j_exp_vec, action='ijk,ijlk->ijl')
            ind_vel_source_global = ind_vel_source_local
            ind_vel_source_global_mat = ind_vel_source_global.reshape((num_nodes, num_points_i, num_panels_s_j, 3))

            # ind_vel_source_local_mat = ind_vel_source_local.reshape((num_nodes, num_points_i, num_panels_s_j, 3))
            AIC_sigma = AIC_sigma.set(csdl.slice[:,start_i:stop_i,start_s_j:stop_s_j,:], value=ind_vel_source_global_mat)

            ind_vel_s_12 = compute_vortex_line_ind_vel(panel_corners_s_j_exp_vec[:,:,0,:],panel_corners_s_j_exp_vec[:,:,1,:],p_eval=eval_pt_w_i_exp_vec[:,:,0,:], mode='wake', vc=1.e-3)
            ind_vel_s_23 = compute_vortex_line_ind_vel(panel_corners_s_j_exp_vec[:,:,1,:],panel_corners_s_j_exp_vec[:,:,2,:],p_eval=eval_pt_w_i_exp_vec[:,:,0,:], mode='wake', vc=1.e-3)
            ind_vel_s_34 = compute_vortex_line_ind_vel(panel_corners_s_j_exp_vec[:,:,2,:],panel_corners_s_j_exp_vec[:,:,3,:],p_eval=eval_pt_w_i_exp_vec[:,:,0,:], mode='wake', vc=1.e-3)
            ind_vel_s_41 = compute_vortex_line_ind_vel(panel_corners_s_j_exp_vec[:,:,3,:],panel_corners_s_j_exp_vec[:,:,0,:],p_eval=eval_pt_w_i_exp_vec[:,:,0,:], mode='wake', vc=1.e-3)

            ind_vel_s = ind_vel_s_12+ind_vel_s_23+ind_vel_s_34+ind_vel_s_41 # (nn, num_interactions, 3)
            ind_vel_s_mat = ind_vel_s.reshape((num_nodes, num_points_i, num_panels_s_j, 3))
            AIC_mu_surf = AIC_mu_surf.set(csdl.slice[:,start_i:stop_i,start_s_j:stop_s_j,:], value=ind_vel_s_mat)

            start_s_j += num_panels_s_j

            '''
            '''

            # ==================== assembling AIC for wake doublets ====================
            nc_w_j, ns_w_j = wake_mesh_dict[surf_name_j]['nc'], mesh_dict[surf_name_j]['ns']
            num_panels_w_j = wake_mesh_dict[surf_name_j]['num_panels']
            stop_w_j += num_panels_w_j

            panel_corners_w_j = wake_mesh_dict[surf_name_j]['panel_corners'][:,t,:,:,:,:]

            num_wake_interactions = num_points_i*num_panels_w_j

            eval_pt_w_i_exp = csdl.expand(eval_pt_i, (num_nodes, nc_i, ns_i, num_panels_w_j, 3), 'ijkl->ijkal')
            eval_pt_w_i_exp_vec = eval_pt_w_i_exp.reshape((num_nodes, num_wake_interactions, 3))

            panel_corners_w_j_exp = csdl.expand(panel_corners_w_j, (num_nodes, num_points_i, nc_w_j-1, ns_w_j-1, 4, 3), 'ijklm->iajklm')
            panel_corners_w_j_exp_vec = panel_corners_w_j_exp.reshape((num_nodes, num_wake_interactions, 4, 3))

            ind_vel_w_12 = compute_vortex_line_ind_vel(panel_corners_w_j_exp_vec[:,:,0,:],panel_corners_w_j_exp_vec[:,:,1,:],p_eval=eval_pt_w_i_exp_vec[:,:,:], mode='wake', vc=1.e-3)
            ind_vel_w_23 = compute_vortex_line_ind_vel(panel_corners_w_j_exp_vec[:,:,1,:],panel_corners_w_j_exp_vec[:,:,2,:],p_eval=eval_pt_w_i_exp_vec[:,:,:], mode='wake', vc=1.e-3)
            ind_vel_w_34 = compute_vortex_line_ind_vel(panel_corners_w_j_exp_vec[:,:,2,:],panel_corners_w_j_exp_vec[:,:,3,:],p_eval=eval_pt_w_i_exp_vec[:,:,:], mode='wake', vc=1.e-3)
            ind_vel_w_41 = compute_vortex_line_ind_vel(panel_corners_w_j_exp_vec[:,:,3,:],panel_corners_w_j_exp_vec[:,:,0,:],p_eval=eval_pt_w_i_exp_vec[:,:,:], mode='wake', vc=1.e-3)

            ind_vel_w = ind_vel_w_12+ind_vel_w_23+ind_vel_w_34+ind_vel_w_41 # (nn, num_interactions, 3)

            ind_vel_w_mat = ind_vel_w.reshape((num_nodes, num_points_i, num_panels_w_j, 3))

            AIC_mu_wake = AIC_mu_wake.set(csdl.slice[:,start_i:stop_i,start_w_j:stop_w_j,:], value=ind_vel_w_mat)

            start_w_j += num_panels_w_j
        start_i += num_points_i 
    
    for nn in csdl.frange(num_nodes):
        for direction in csdl.frange(3):
            sigma_induced_vel = csdl.matvec(AIC_sigma[nn,:,:,direction], sigma[nn,t,:])

            mu_surf_induced_vel = csdl.matvec(AIC_mu_surf[nn,:,:,direction], mu[nn,t,:])
            mu_wake_induced_vel = csdl.matvec(AIC_mu_wake[nn,:,:,direction], mu_wake[nn,t,:])

            total_induced_vel = sigma_induced_vel + mu_surf_induced_vel + mu_wake_induced_vel
            # induced_vel = induced_vel.set(csdl.slice[nn,:,direction], value=sigma_induced_vel)
            induced_vel = induced_vel.set(csdl.slice[nn,:,direction], value=total_induced_vel)
            # induced_vel = induced_vel.set(csdl.slice[nn,:,direction], value=mu_wake_induced_vel)
            # induced_vel = induced_vel.set(csdl.slice[nn,:,direction], value=mu_surf_induced_vel)

    return induced_vel


'''
NOTES:
- two for loops (one nested)
- outer loop scans through surface wakes (the evaluation points are the wake coordinates)
- inner loop scans wakes and bodies to get influence from doublets/sources
'''