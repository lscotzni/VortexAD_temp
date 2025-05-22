import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.panel_method.steady.steady_source_doublet_solver import source_doublet_solver
from VortexAD.core.panel_method.steady.higher_order.linear_doublet_solver import linear_doublet_solver

# def steady_panel_solver(mesh_list, mesh_velocity_list, patches=False, coll_vel_list=False):
def steady_panel_solver(*args, rho=1.225, mesh_mode='structured', batch_size=None, 
                        Cp_cutoff=-100., patches=False, higher_order=False, boundary_condition='Dirichlet', 
                        iterative=False, ROM=False):
    '''
    mesh_list: list of lists
        - each entry represents a surface
            - each entry in the sublist represents a patch
    mesh_velocity_list: list of lists
        - same structure as mesh_list, just representing velocities
    patches: list of lists
        - each entry represents the wake shedding representation of each surface
            - None: does not shed a wake
            - wrap: TE is inferred from the grid structure
                - implies [-1,:] - [0,:] structure for kutta condition (chordwise)
            - ['upper', 'lower']: tells us which patch is upper and lower
                - assumes grid ordering is from nose to tail
                - for kutta condition, we take final entry of both patches 

    '''
    patch_flag = False # done by default
    if mesh_mode == 'structured':
        mesh_list = args[0]
        mesh_velocity_list = args[1]
        try:
            coll_vel_list = args[2]
            coll_vel = True
        except:
            coll_vel = False

        exp_orig_mesh_dict = {}
        num_surfaces = len(mesh_list)
        surface_counter = 0
        
        if not patches: # NO PATCHES
            for i in range(num_surfaces): # surface loop
                surface_name = f'surface_{surface_counter}'

                surf_dict = {}
                surf_dict['mesh'] = mesh_list[i]
                surf_dict['nodal_velocity'] = mesh_velocity_list[i] * -1.
                if coll_vel:
                    surf_dict['coll_point_velocity'] = coll_vel_list[i] * -1. 
                else:
                    surf_dict['coll_point_velocity'] = coll_vel
                exp_orig_mesh_dict[surface_name] = surf_dict

                if i == 0:
                    num_nodes = mesh_list[i].shape[0] # NOTE: CHECK THIS LINE
                
                surface_counter += 1
        else: # YES PATCHES (deprecated)
            patch_flag = True
            patch_counter = 0
            for i in range(num_surfaces): # surface loop
                surface_name = f'surface_{surface_counter}'
                exp_orig_mesh_dict[surface_name] = {}
                surf = mesh_list[i]
                num_surf_patches = len(surf)
                for j in range(num_surf_patches): # patch loop
                    patch_dict = {}
                    patch_name = f'patch_{patch_counter}'
                    patch_dict['mesh'] = mesh_list[i][j]
                    patch_dict['patch'] = patches[i][j]
                    # exp_orig_mesh_dict[surface_name]['mesh'] = mesh_list[i]
                    patch_dict['nodal_velocity'] = mesh_velocity_list[i][j] * -1.
                    # exp_orig_mesh_dict[surface_name]['nodal_velocity'] = mesh_velocity_list[i] * -1. 
                    if coll_vel_list:
                        # exp_orig_mesh_dict[surface_name]['coll_point_velocity'] = coll_vel_list[i] * -1. 
                        patch_dict['coll_point_velocity'] = coll_vel_list[i] * -1. 
                    else:
                        # exp_orig_mesh_dict[surface_name]['coll_point_velocity'] = coll_vel_list
                        patch_dict['coll_point_velocity'] = coll_vel
                    patch_counter += 1
                    exp_orig_mesh_dict[surface_name][patch_name] = patch_dict

                # exp_orig_mesh_dict[surface_name]['patches'] = patches[i]
                if i == 0:
                    num_nodes = mesh_list[i][0].shape[0] # NOTE: CHECK THIS LINE
                
                surface_counter += 1

    elif mesh_mode == 'unstructured':
        points = args[0]
        cells, cell_adjacency, points2cells = args[1][0], args[1][1], args[1][2]
        TE_node_indices, TE_edges, TE_cells = args[2]
        upper_TE_cells, lower_TE_cells = TE_cells[0], TE_cells[1]
        point_velocity = args[3]

        num_nodes = points.shape[0]

        exp_orig_mesh_dict = {}
        exp_orig_mesh_dict['points'] = points
        exp_orig_mesh_dict['nodal_velocity'] = point_velocity * -1.
        exp_orig_mesh_dict['cell_point_indices'] = cells
        exp_orig_mesh_dict['cell_adjacency'] = cell_adjacency
        exp_orig_mesh_dict['points2cells'] = points2cells

        exp_orig_mesh_dict['TE_node_indices'] = TE_node_indices
        exp_orig_mesh_dict['TE_edges'] = TE_edges
        exp_orig_mesh_dict['upper_TE_cells'] = upper_TE_cells
        exp_orig_mesh_dict['lower_TE_cells'] = lower_TE_cells

    else:
        raise ValueError(
            'Invalid input for mesh mode. Options are structured (including patches) or unstructured'
        )
    if higher_order and not patch_flag:
        outputs = linear_doublet_solver(exp_orig_mesh_dict, num_nodes, mesh_mode, rho)
        output_dict = outputs[0]
        mesh_dict = outputs[1]
        mu = outputs[2]
        sigma = outputs[3]

        return output_dict, mesh_dict, mu, sigma
    
    else:
        # sigma = source_doublet_solver(exp_orig_mesh_dict, num_nodes, mesh_mode, rho, boundary_condition, patch_flag)
        # return sigma

        # AIC_mu = source_doublet_solver(exp_orig_mesh_dict, num_nodes, mesh_mode, rho, boundary_condition, patch_flag)
        # return AIC_mu

        # mu = source_doublet_solver(exp_orig_mesh_dict, num_nodes, mesh_mode, rho, boundary_condition, patch_flag)
        # return mu

        outputs = source_doublet_solver(exp_orig_mesh_dict, num_nodes, mesh_mode, rho, batch_size, Cp_cutoff, boundary_condition, patch_flag, iterative, ROM)
        output_dict = outputs[0]
        mesh_dict = outputs[1]
        mu = outputs[2]
        sigma = outputs[3]

        return output_dict, mesh_dict, mu, sigma