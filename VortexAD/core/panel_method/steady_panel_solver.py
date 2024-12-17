import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.panel_method.steady.steady_source_doublet_solver import source_doublet_solver

def steady_panel_solver(mesh_list, mesh_velocity_list, patches=False, coll_vel_list=False):
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
    
    exp_orig_mesh_dict = {}
    num_surfaces = len(mesh_list)
    surface_counter = 0
    
    if not patches: # NO PATCHES
        patch_flag = False
        for i in range(num_surfaces): # surface loop
            surface_name = f'surface_{surface_counter}'

            surf_dict = {}
            surf_dict['mesh'] = mesh_list[i]
            surf_dict['nodal_velocity'] = mesh_velocity_list[i] * -1.
            if coll_vel_list:
                surf_dict['coll_point_velocity'] = coll_vel_list[i] * -1. 
            else:
                surf_dict['coll_point_velocity'] = coll_vel_list
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
                    patch_dict['coll_point_velocity'] = coll_vel_list
                patch_counter += 1
                exp_orig_mesh_dict[surface_name][patch_name] = patch_dict

            # exp_orig_mesh_dict[surface_name]['patches'] = patches[i]
            if i == 0:
                num_nodes = mesh_list[i][0].shape[0] # NOTE: CHECK THIS LINE
            
            surface_counter += 1

    outputs = source_doublet_solver(exp_orig_mesh_dict, num_nodes, patch_flag)
    output_dict = outputs[0]
    mesh_dict = outputs[1]
    mu = outputs[2]
    sigma = outputs[3]
    wake_dict = outputs[4]

    return output_dict, mesh_dict, mu, sigma, wake_dict