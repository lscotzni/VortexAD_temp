import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.panel_method.steady.steady_source_doublet_solver import source_doublet_solver

def steady_panel_solver(mesh_list, mesh_velocity_list, coll_vel_list=False):

    exp_orig_mesh_dict = {}
    surface_counter = 0
    for i in range(len(mesh_list)):
        surface_name = f'surface_{surface_counter}'
        exp_orig_mesh_dict[surface_name] = {}
        exp_orig_mesh_dict[surface_name]['mesh'] = mesh_list[i]
        exp_orig_mesh_dict[surface_name]['nodal_velocity'] = mesh_velocity_list[i] * -1. 
        if coll_vel_list:
            exp_orig_mesh_dict[surface_name]['coll_point_velocity'] = coll_vel_list[i] * -1. 
        else:
            exp_orig_mesh_dict[surface_name]['coll_point_velocity'] = coll_vel_list
        if i == 0:
            num_nodes = mesh_list[i].shape[0] # NOTE: CHECK THIS LINE
        
        surface_counter += 1

    outputs = source_doublet_solver(exp_orig_mesh_dict, num_nodes)
    output_dict = outputs[0]
    mesh_dict = outputs[1]
    mu = outputs[2]
    sigma = outputs[3]

    return output_dict, mesh_dict, mu, sigma