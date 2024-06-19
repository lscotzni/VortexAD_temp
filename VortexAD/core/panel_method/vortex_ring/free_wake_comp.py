import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.panel_method.vortex_ring.vortex_line_functions import compute_vortex_line_ind_vel

def free_wake_comp(num_nodes, t, mesh_dict, mu, wake_mesh_dict, mu_wake):
    surface_names = list(mesh_dict.keys())
    num_wake_nodes = 0
    num_wake_panels = 0
    num_surf_panels = 0
    for surface in surface_names:
        nc_w, ns_w = wake_mesh_dict[surface]['mesh'].shape[-3], wake_mesh_dict[surface]['mesh'].shape[-2]
        num_wake_nodes += nc_w*ns_w
        
        num_wake_panels += wake_mesh_dict[surface]['num_panels']

        num_surf_panels += mesh_dict[surface]['num_panels']

    AIC_gamma_surf = csdl.Variable(shape=(num_nodes, num_wake_nodes, num_surf_panels, 3), value=0.)
    AIC_gamma_wake = csdl.Variable(shape=(num_nodes, num_wake_nodes, num_wake_panels, 3), value=0.)

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

            num_surf_interactions = num_points_i*num_panels_s_j

            eval_pt_w_i_exp = csdl.expand(eval_pt_i, (num_nodes, nc_i, ns_i, num_panels_s_j, 3), 'ijkl->ijkal')
            eval_pt_w_i_exp_vec = eval_pt_w_i_exp.reshape((num_nodes, num_surf_interactions, 3))

            panel_corners_s_j_exp = csdl.expand(panel_corners_s_j, (num_nodes, num_points_i, nc_s_j-1, ns_s_j-1, 4, 3), 'ijklm->iajklm')
            panel_corners_s_j_exp_vec = panel_corners_s_j_exp.reshape((num_nodes, num_surf_interactions, 4, 3))

            ind_vel_s_12 = compute_vortex_line_ind_vel(panel_corners_s_j_exp_vec[:,:,0,:],panel_corners_s_j_exp_vec[:,:,1,:],p_eval=eval_pt_w_i_exp_vec[:,:,:], mode='wake')
            ind_vel_s_23 = compute_vortex_line_ind_vel(panel_corners_s_j_exp_vec[:,:,1,:],panel_corners_s_j_exp_vec[:,:,2,:],p_eval=eval_pt_w_i_exp_vec[:,:,:], mode='wake')
            ind_vel_s_34 = compute_vortex_line_ind_vel(panel_corners_s_j_exp_vec[:,:,2,:],panel_corners_s_j_exp_vec[:,:,3,:],p_eval=eval_pt_w_i_exp_vec[:,:,:], mode='wake')
            ind_vel_s_41 = compute_vortex_line_ind_vel(panel_corners_s_j_exp_vec[:,:,3,:],panel_corners_s_j_exp_vec[:,:,0,:],p_eval=eval_pt_w_i_exp_vec[:,:,:], mode='wake')

            ind_vel_s = ind_vel_s_12+ind_vel_s_23+ind_vel_s_34+ind_vel_s_41 # (nn, num_interactions, 3)
            ind_vel_s_mat = ind_vel_s.reshape((num_nodes, num_points_i, num_panels_s_j, 3))
            AIC_gamma_surf = AIC_gamma_surf.set(csdl.slice[:,start_i:stop_i,start_s_j:stop_s_j,:], value=ind_vel_s_mat)

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

            ind_vel_w_12 = compute_vortex_line_ind_vel(panel_corners_w_j_exp_vec[:,:,0,:],panel_corners_w_j_exp_vec[:,:,1,:],p_eval=eval_pt_w_i_exp_vec[:,:,:], mode='wake')
            ind_vel_w_23 = compute_vortex_line_ind_vel(panel_corners_w_j_exp_vec[:,:,1,:],panel_corners_w_j_exp_vec[:,:,2,:],p_eval=eval_pt_w_i_exp_vec[:,:,:], mode='wake')
            ind_vel_w_34 = compute_vortex_line_ind_vel(panel_corners_w_j_exp_vec[:,:,2,:],panel_corners_w_j_exp_vec[:,:,3,:],p_eval=eval_pt_w_i_exp_vec[:,:,:], mode='wake')
            ind_vel_w_41 = compute_vortex_line_ind_vel(panel_corners_w_j_exp_vec[:,:,3,:],panel_corners_w_j_exp_vec[:,:,0,:],p_eval=eval_pt_w_i_exp_vec[:,:,:], mode='wake')

            # ind_vel_w_12 = compute_vortex_line_ind_vel(panel_corners_w_j_exp_vec[:,:,1,:],panel_corners_w_j_exp_vec[:,:,0,:],p_eval=eval_pt_w_i_exp_vec[:,:,:])
            # ind_vel_w_23 = compute_vortex_line_ind_vel(panel_corners_w_j_exp_vec[:,:,2,:],panel_corners_w_j_exp_vec[:,:,1,:],p_eval=eval_pt_w_i_exp_vec[:,:,:])
            # ind_vel_w_34 = compute_vortex_line_ind_vel(panel_corners_w_j_exp_vec[:,:,3,:],panel_corners_w_j_exp_vec[:,:,2,:],p_eval=eval_pt_w_i_exp_vec[:,:,:])
            # ind_vel_w_41 = compute_vortex_line_ind_vel(panel_corners_w_j_exp_vec[:,:,0,:],panel_corners_w_j_exp_vec[:,:,3,:],p_eval=eval_pt_w_i_exp_vec[:,:,:])

            ind_vel_w = ind_vel_w_12+ind_vel_w_23+ind_vel_w_34+ind_vel_w_41 # (nn, num_interactions, 3)

            ind_vel_w_mat = ind_vel_w.reshape((num_nodes, num_points_i, num_panels_w_j, 3))

            AIC_gamma_wake = AIC_gamma_wake.set(csdl.slice[:,start_i:stop_i,start_w_j:stop_w_j,:], value=ind_vel_w_mat)

            start_w_j += num_panels_w_j
        start_i += num_points_i
    
    for nn in csdl.frange(num_nodes):
        for direction in csdl.frange(3):
            gamma_surf_induced_vel = csdl.matvec(AIC_gamma_surf[nn,:,:,direction], mu[nn,t,:])
            gamma_wake_induced_vel = csdl.matvec(AIC_gamma_wake[nn,:,:,direction], mu_wake[nn,t,:])
            induced_vel = induced_vel.set(csdl.slice[nn,:,direction], value=gamma_surf_induced_vel+gamma_wake_induced_vel)
            # induced_vel = induced_vel.set(csdl.slice[nn,:,dir], value=mu_wake_induced_vel)

    return induced_vel


'''
NOTES:
- two for loops (one nested)
- outer loop scans through surface wakes (the evaluation points are the wake coordinates)
- inner loop scans wakes and bodies to get influence from doublets/sources
'''