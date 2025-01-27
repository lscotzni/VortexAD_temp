import csdl_alpha as csdl

# NO PATCHES
from VortexAD.core.panel_method.steady.pre_processor import pre_processor
from VortexAD.core.panel_method.steady.mu_sigma_solver import mu_sigma_solver
from VortexAD.core.panel_method.steady.post_processor import post_processor, unstructured_post_processor

# YES PATCHES
from VortexAD.core.panel_method.steady.pre_processor_patches import pre_processor_patches
from VortexAD.core.panel_method.steady.mu_sigma_solver_patches import mu_sigma_solver_patches
from VortexAD.core.panel_method.steady.post_processor_patches import post_processor_patches

def source_doublet_solver(exp_orig_mesh_dict, num_nodes, mesh_mode, patch_flag):
    if not patch_flag: # both structured and unstructured grids
        print('running pre-processing')
        mesh_dict = pre_processor(exp_orig_mesh_dict, mode=mesh_mode)

        print('solving for doublet strengths')
        mu, sigma, wake_dict, AIC_mu, AIC_sigma = mu_sigma_solver(num_nodes, mesh_dict, mode=mesh_mode)

        print('running post-processor')
        if mesh_mode == 'structured':
            output_dict = post_processor(mesh_dict, mu, sigma, num_nodes)
        elif mesh_mode == 'unstructured':
            output_dict = unstructured_post_processor(mesh_dict, mu, sigma, num_nodes)

        output_dict['wake_dict'] = wake_dict
        output_dict['AIC_mu'] = AIC_mu
        output_dict['AIC_sigma'] = AIC_sigma

    elif patch_flag: # only structured sub grids
        print('running pre-processing')
        mesh_dict = pre_processor_patches(exp_orig_mesh_dict)

        print('solving for doublet strengths')
        mu, sigma, wake_dict = mu_sigma_solver_patches(num_nodes, mesh_dict)

        print('running post-processor')
        output_dict = post_processor_patches(mesh_dict, mu, sigma, num_nodes)


    return output_dict, mesh_dict, mu, sigma