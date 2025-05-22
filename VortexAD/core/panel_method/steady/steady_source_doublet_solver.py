import csdl_alpha as csdl

# NO PATCHES
from VortexAD.core.panel_method.steady.pre_processor import pre_processor
from VortexAD.core.panel_method.steady.mu_sigma_solver import mu_sigma_solver
from VortexAD.core.panel_method.steady.mu_sigma_solver_iterative import mu_sigma_solver_iterative
from VortexAD.core.panel_method.steady.post_processor import post_processor, unstructured_post_processor

# YES PATCHES
from VortexAD.core.panel_method.steady.pre_processor_patches import pre_processor_patches
from VortexAD.core.panel_method.steady.mu_sigma_solver_patches import mu_sigma_solver_patches
from VortexAD.core.panel_method.steady.post_processor_patches import post_processor_patches

def source_doublet_solver(exp_orig_mesh_dict, num_nodes, mesh_mode, rho, batch_size, Cp_cutoff, boundary_condition, patch_flag, iterative=False, ROM=False):
    if not patch_flag: # both structured and unstructured grids
        print('running pre-processing')
        mesh_dict = pre_processor(exp_orig_mesh_dict, mode=mesh_mode)

        print('solving for doublet strengths')
        if not iterative:
            # sigma = mu_sigma_solver(num_nodes, mesh_dict, mode=mesh_mode, bc=boundary_condition)
            # return sigma
        
            # AIC_mu = mu_sigma_solver(num_nodes, mesh_dict, mode=mesh_mode, bc=boundary_condition)
            # return AIC_mu
        
            # mu, sigma, wake_dict, AIC_mu, AIC_sigma, AIC_mu_orig = mu_sigma_solver(num_nodes, mesh_dict, mode=mesh_mode, bc=boundary_condition)
            mu, sigma, wake_dict, AIC_mu, AIC_sigma, RHS = mu_sigma_solver(num_nodes, mesh_dict, mode=mesh_mode, batch_size=batch_size, bc=boundary_condition, ROM=ROM)
        else:
            mu, sigma, wake_dict = mu_sigma_solver_iterative(num_nodes, mesh_dict, mode=mesh_mode, batch_size=batch_size, bc=boundary_condition, ROM=ROM)
        # return mu
        print('running post-processor')
        if mesh_mode == 'structured':
            output_dict = post_processor(mesh_dict, mu, sigma, num_nodes, rho, Cp_cutoff)
        elif mesh_mode == 'unstructured':
            output_dict = unstructured_post_processor(mesh_dict, mu, sigma, num_nodes, rho, Cp_cutoff)

        output_dict['wake_dict'] = wake_dict
        output_dict['AIC_mu'] = AIC_mu
        # output_dict['AIC_mu_orig'] = AIC_mu_orig
        output_dict['AIC_sigma'] = AIC_sigma
        output_dict['RHS'] = RHS

    elif patch_flag: # only structured sub grids
        print('running pre-processing')
        mesh_dict = pre_processor_patches(exp_orig_mesh_dict)

        print('solving for doublet strengths')
        mu, sigma, wake_dict = mu_sigma_solver_patches(num_nodes, mesh_dict)

        print('running post-processor')
        output_dict = post_processor_patches(mesh_dict, mu, sigma, num_nodes, rho)


    return output_dict, mesh_dict, mu, sigma