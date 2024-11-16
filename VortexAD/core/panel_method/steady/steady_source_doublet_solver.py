import csdl_alpha as csdl

from VortexAD.core.panel_method.steady.pre_processor import pre_processor
from VortexAD.core.panel_method.steady.mu_sigma_solver import mu_sigma_solver
from VortexAD.core.panel_method.steady.post_processor import post_processor

def source_doublet_solver(exp_orig_mesh_dict, num_nodes):
    print('running pre-processing')
    with csdl.namespace('pre-processing'):
        mesh_dict = pre_processor(exp_orig_mesh_dict)

    print('solving for doublet strengths')
    mu, sigma = mu_sigma_solver(num_nodes, mesh_dict)

    print('running post-processor')
    output_dict = post_processor(mesh_dict, mu, sigma, num_nodes)

    return output_dict, mesh_dict, mu, sigma