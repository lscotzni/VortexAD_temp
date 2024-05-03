import numpy as np 
import csdl_alpha as csdl 

# from VortexAD.core.parse_ac_states import parse_ac_states
from VortexAD.core.vlm.pre_processor import pre_processor
from VortexAD.core.vlm.gamma_solver import gamma_solver
from VortexAD.core.vlm.post_processor import post_processor

class VLMSolver(object):
    def __init__(self):
        pass



def vlm_solver(orig_mesh_dict, ac_states):
    '''
    VLM solver (add description)
    '''
    exp_orig_mesh_dict = {}
    for key in orig_mesh_dict:
        exp_orig_mesh_dict[key] = {}
        exp_orig_mesh_dict[key]['mesh'] = orig_mesh_dict[key]
        num_nodes = orig_mesh_dict[key].shape[0] # NOTE: CHECK THIS LINE

    # parse ac_states
    # u, v, w, alpha = parse_ac_states(ac_states)
    alpha = 1.
    V_inf = csdl.Variable(value=np.array([10., 0., 0.]))
    mesh_dict = pre_processor(exp_orig_mesh_dict, alpha)

    gamma = gamma_solver(num_nodes, mesh_dict, V_inf, alpha)

    output_dict = post_processor(num_nodes, mesh_dict, gamma, V_inf, alpha)

    return output_dict