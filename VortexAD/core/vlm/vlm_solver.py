import numpy as np 
import csdl_alpha as csdl 

# from VortexAD.core.parse_ac_states import parse_ac_states
from VortexAD.core.vlm.pre_processor import pre_processor
from VortexAD.core.vlm.gamma_solver import gamma_solver
from VortexAD.core.vlm.post_processor import post_processor

class VLMSolver(object):
    def __init__(self):
        pass



def vlm_solver(orig_mesh_dict, ac_states, V_inf, alpha):
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
    print('running pre-processing')
    mesh_dict = pre_processor(exp_orig_mesh_dict)
    V_inf_rot = csdl.Variable(value=np.zeros((num_nodes,3)))
    alpha = csdl.Variable(value=alpha)
    V_inf = csdl.Variable(value=V_inf)
    V_rot_mat = csdl.Variable(value=np.zeros((num_nodes, 3,3)))
    V_rot_mat = V_rot_mat.set(csdl.slice[:,1,1], value=1.)
    V_rot_mat = V_rot_mat.set(csdl.slice[:,0,0], value=csdl.cos(alpha))
    V_rot_mat = V_rot_mat.set(csdl.slice[:,2,2], value=csdl.cos(alpha))
    V_rot_mat = V_rot_mat.set(csdl.slice[:,2,0], value=csdl.sin(alpha))
    V_rot_mat = V_rot_mat.set(csdl.slice[:,0,2], value=-csdl.sin(alpha))
    for i in csdl.frange(num_nodes):
        # V_rot_mat = np.zeros((3,3))
        # V_rot_mat[1,1] = 1.
        # V_rot_mat[0,0] = V_rot_mat[2,2] = csdl.cos(alpha[[i]])
        # V_rot_mat[2,0] = csdl.sin(alpha[i])
        # V_rot_mat[0,2] = -csdl.sin(alpha[i])
        V_inf_rot = V_inf_rot.set(csdl.slice[i,:], value=csdl.matvec(V_rot_mat[i,:,:], V_inf[i,:]))

    print('solving for circulation strengths')
    gamma = gamma_solver(num_nodes, mesh_dict, V_inf_rot)

    print('running post-processing')
    output_dict = post_processor(num_nodes, mesh_dict, gamma, V_inf_rot, alpha)

    return output_dict