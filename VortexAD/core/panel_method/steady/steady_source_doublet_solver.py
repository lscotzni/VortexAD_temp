import csdl_alpha as csdl

# NO PATCHES
from VortexAD.core.panel_method.steady.pre_processor import pre_processor
from VortexAD.core.panel_method.steady.mu_sigma_solver import mu_sigma_solver
from VortexAD.core.panel_method.steady.post_processor import post_processor

# YES PATCHES
from VortexAD.core.panel_method.steady.pre_processor_patches import pre_processor_patches
from VortexAD.core.panel_method.steady.mu_sigma_solver_patches import mu_sigma_solver_patches
from VortexAD.core.panel_method.steady.post_processor_patches import post_processor_patches

def source_doublet_solver(exp_orig_mesh_dict, num_nodes, patch_flag):
    if not patch_flag:
        print('running pre-processing')
        mesh_dict = pre_processor(exp_orig_mesh_dict)

        print('solving for doublet strengths')
        mu, sigma, wake_dict = mu_sigma_solver(num_nodes, mesh_dict)

        print('running post-processor')
        output_dict = post_processor(mesh_dict, mu, sigma, num_nodes)

    elif patch_flag:
        print('running pre-processing')
        mesh_dict = pre_processor_patches(exp_orig_mesh_dict)

        print('solving for doublet strengths')
        mu, sigma, wake_dict = mu_sigma_solver_patches(num_nodes, mesh_dict)

        print('running post-processor')
        output_dict = post_processor_patches(mesh_dict, mu, sigma, num_nodes)


    return output_dict, mesh_dict, mu, sigma, wake_dict