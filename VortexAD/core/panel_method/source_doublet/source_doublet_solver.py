import csdl_alpha as csdl

from VortexAD.core.panel_method.source_doublet.pre_processor import pre_processor
from VortexAD.core.panel_method.source_doublet.pre_processor_new import pre_processor_new
from VortexAD.core.panel_method.source_doublet.mu_sigma_solver import mu_sigma_solver
from VortexAD.core.panel_method.source_doublet.post_processor import post_processor, unstructured_post_processor
from VortexAD.core.boundary_layer.boundary_layer_solver import boundary_layer_solver

def source_doublet_solver(exp_orig_mesh_dict, num_nodes, nt, dt, mesh_mode, free_wake, boundary_layer, boundary_layer_coupling):
    print('running pre-processing')
    with csdl.namespace('pre-processing'):
        # mesh_dict = pre_processor(exp_orig_mesh_dict, mode=mesh_mode)
        mesh_dict = pre_processor_new(exp_orig_mesh_dict, mode=mesh_mode)

    print('solving for doublet strengths and propagating the wake')
    mu, sigma, mu_wake, wake_mesh_dict = mu_sigma_solver(num_nodes, 
        nt, 
        mesh_dict, 
        dt, 
        mesh_mode=mesh_mode, 
        free_wake=free_wake, 
        boundary_layer_coupling=boundary_layer_coupling
    )

    if mesh_mode == 'structured':
        output_dict = post_processor(mesh_dict, mu, sigma, num_nodes, nt, dt)
    elif mesh_mode == 'unstructured':
        output_dict = unstructured_post_processor(mesh_dict, mu, sigma, num_nodes, nt, dt)

    if boundary_layer:
        delta_star, theta, H, Cf = boundary_layer_solver(mesh_dict, output_dict, boundary_layer, num_nodes, nt, dt)
        BL_outputs = {
            'delta_star': delta_star,
            'theta': theta,
            'H': H,
            'Cf': Cf
        }
        return output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake, BL_outputs
    else:

        return output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake