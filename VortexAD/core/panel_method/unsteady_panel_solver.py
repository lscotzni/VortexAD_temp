import numpy as np 
import csdl_alpha as csdl 

from VortexAD.core.panel_method.pre_processor import pre_processor
from VortexAD.core.panel_method.mu_solver import mu_solver

def unsteady_panel_solver(mesh_list, mesh_velocity_list):

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

    print('running pre-processing')
    mesh_dict = pre_processor(exp_orig_mesh_dict)

    print('solving for doublet strengths and propagating the wake')
    mu = mu_solver(num_nodes, nt, mesh_dict)






    return mesh_dict