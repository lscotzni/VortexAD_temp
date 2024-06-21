import numpy as np 
import csdl_alpha as csdl 


from VortexAD.core.panel_method.source_doublet.source_doublet_solver import source_doublet_solver
from VortexAD.core.panel_method.vortex_ring.vortex_ring_solver import vortex_ring_solver


# from VortexAD.core.panel_method.pre_processor import pre_processor
# from VortexAD.core.panel_method.mu_sigma_solver import mu_sigma_solver
# from VortexAD.core.panel_method.vortex_ring_solver import vortex_ring_solver
# from VortexAD.core.panel_method.post_processor import post_processor

def unsteady_panel_solver(mesh_list, mesh_velocity_list, dt, mesh_mode='structured', mode='source-doublet', connectivity=None, free_wake=False):
    '''
    2 modes:
    - source doublet (Dirichlet (no-perturbation potential in the body))
    - vortex rings (Neumann (no-penetration condition))
    '''
    exp_orig_mesh_dict = {}
    surface_counter = 0
    for i in range(len(mesh_list)):
        surface_name = f'surface_{surface_counter}'
        exp_orig_mesh_dict[surface_name] = {}
        exp_orig_mesh_dict[surface_name]['mesh'] = mesh_list[i]
        exp_orig_mesh_dict[surface_name]['nodal_velocity'] = mesh_velocity_list[i] * -1. 
        if i == 0:
            num_nodes = mesh_list[i].shape[0] # NOTE: CHECK THIS LINE
            nt = mesh_list[i].shape[1]
        
        surface_counter += 1


    if mode == 'source-doublet':
        outputs = source_doublet_solver(exp_orig_mesh_dict, num_nodes, nt, dt, mesh_mode, connectivity, free_wake)
        output_dict = outputs[0]
        mesh_dict = outputs[1]
        wake_mesh_dict = outputs[2]
        mu = outputs[3]
        sigma = outputs[4]
        mu_wake = outputs[5]

        return output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake

    elif mode == 'vortex-ring':
        outputs = vortex_ring_solver(exp_orig_mesh_dict, num_nodes, nt, dt, mesh_mode, connectivity, free_wake)
        output_dict = outputs[0]
        mesh_dict = outputs[1]
        wake_mesh_dict = outputs[2]
        mu = outputs[3]
        mu_wake = outputs[4]

        return output_dict, mesh_dict, wake_mesh_dict, mu, mu_wake

    else:
        raise TypeError('Invalid solver mode. Options are source-doublet or vortex-ring')
    


    # print('running pre-processing')
    # with csdl.namespace('pre-processing'):
    #     mesh_dict = pre_processor(exp_orig_mesh_dict, mode=mesh_mode, connectivity=connectivity)

    # print('solving for doublet strengths and propagating the wake')
    # if mode == 'source_doublet':
    #     mu, sigma, mu_wake, wake_mesh_dict = mu_sigma_solver(num_nodes, nt, mesh_dict, dt, free_wake=free_wake)
    # elif mode == 'vortex_ring':
    #     sigma = None
    #     mu, mu_wake, wake_mesh_dict = vortex_ring_solver(num_nodes, nt, mesh_dict, dt, free_wake=free_wake)
    #     pass
    # with csdl.namespace('post-processing'):
    #     output_dict = post_processor(mesh_dict, mu, sigma, num_nodes, nt, dt)

    