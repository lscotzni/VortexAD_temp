import numpy as np 
import csdl_alpha as csdl 

def induced_velocity_force_pts(num_nodes, mesh_dict, gamma_vec):
    '''
    Compute induced velocities of the field at the force evaluation points
    v_i = AIC*gamma
    '''
    return