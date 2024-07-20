import csdl_alpha as csdl

from VortexAD.core.panel_method.source_doublet.pre_processor import pre_processor
from VortexAD.core.panel_method.source_doublet.mu_sigma_solver import mu_sigma_solver
from VortexAD.core.panel_method.source_doublet.post_processor import post_processor, unstructured_post_processor

def source_doublet_solver(exp_orig_mesh_dict, num_nodes, nt, dt, mesh_mode, free_wake):
    print('running pre-processing')
    with csdl.namespace('pre-processing'):
        mesh_dict = pre_processor(exp_orig_mesh_dict, mode=mesh_mode)

    print('solving for doublet strengths and propagating the wake')
    mu, sigma, mu_wake, wake_mesh_dict = mu_sigma_solver(num_nodes, nt, mesh_dict, dt, mesh_mode=mesh_mode, free_wake=free_wake)

    if mesh_mode == 'structured':
        output_dict = post_processor(mesh_dict, mu, sigma, num_nodes, nt, dt)
    elif mesh_mode == 'unstructured':
        output_dict = unstructured_post_processor(mesh_dict, mu, sigma, num_nodes, nt, dt)

    return output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake