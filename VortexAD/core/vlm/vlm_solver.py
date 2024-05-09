import numpy as np 
import csdl_alpha as csdl 

# from VortexAD.core.parse_ac_states import parse_ac_states
from VortexAD.core.vlm.pre_processor import pre_processor
from VortexAD.core.vlm.gamma_solver import gamma_solver
from VortexAD.core.vlm.post_processor import post_processor

class VLMSolver(object):
    def __init__(self):
        pass

'''
inputs:
- list of meshes
- list of mesh velocities

MeshParameters
- .nodal_coordinates
- .nodal_velocities

AVOID asdict() method in data classes (VERY SLOW)


add surface 1, surface 2, etc. within the function
'''

# def vlm_solver(orig_mesh_dict, V_inf, alpha):
def vlm_solver(mesh_list, mesh_velocity_list):
    '''
    VLM solver (add description)
    '''
    exp_orig_mesh_dict = {}
    surface_counter = 0

    for i in range(len(mesh_list)):
        surface_name = f'surface_{surface_counter}'
        exp_orig_mesh_dict[surface_name] = {}
        exp_orig_mesh_dict[surface_name]['mesh'] = mesh_list[i]
        exp_orig_mesh_dict[surface_name]['nodal_velocity'] = mesh_velocity_list[i]

        num_nodes = mesh_list[i].shape[0] # NOTE: CHECK THIS LINE
        surface_counter += 1

    # parse ac_states
    # u, v, w, alpha = parse_ac_states(ac_states)
    print('running pre-processing')
    mesh_dict = pre_processor(exp_orig_mesh_dict)

    # V_inf_rot = csdl.Variable(value=np.zeros((num_nodes,3)))
    # alpha = csdl.Variable(value=alpha)
    # V_inf = csdl.Variable(value=V_inf)
    # V_rot_mat = csdl.Variable(value=np.zeros((num_nodes, 3,3)))
    # V_rot_mat = V_rot_mat.set(csdl.slice[:,1,1], value=1.)
    # V_rot_mat = V_rot_mat.set(csdl.slice[:,0,0], value=csdl.cos(alpha))
    # V_rot_mat = V_rot_mat.set(csdl.slice[:,2,2], value=csdl.cos(alpha))
    # V_rot_mat = V_rot_mat.set(csdl.slice[:,2,0], value=csdl.sin(alpha))
    # V_rot_mat = V_rot_mat.set(csdl.slice[:,0,2], value=-csdl.sin(alpha))
    # for i in csdl.frange(num_nodes):
    #     # V_rot_mat = np.zeros((3,3))
    #     # V_rot_mat[1,1] = 1.
    #     # V_rot_mat[0,0] = V_rot_mat[2,2] = csdl.cos(alpha[[i]])
    #     # V_rot_mat[2,0] = csdl.sin(alpha[i])
    #     # V_rot_mat[0,2] = -csdl.sin(alpha[i])
    #     V_inf_rot = V_inf_rot.set(csdl.slice[i,:], value=csdl.matvec(V_rot_mat[i,:,:], V_inf[i,:]))

    print('solving for circulation strengths')
    gamma = gamma_solver(num_nodes, mesh_dict)

    print('running post-processing')
    surface_output_dict, total_output_dict = post_processor(num_nodes, mesh_dict, gamma)

    output_vg = csdl.VariableGroup()
    output_vg.total_lift = total_output_dict['total_lift']
    output_vg.total_drag = total_output_dict['total_drag']
    output_vg.total_force = total_output_dict['total_force']
    output_vg.total_moment = total_output_dict['total_moment']

    output_vg.surface_CL = surface_output_dict['surface_CL']
    output_vg.surface_CDi = surface_output_dict['surface_CDi']
    output_vg.surface_lift = surface_output_dict['surface_lift']
    output_vg.surface_drag = surface_output_dict['surface_drag']
    output_vg.surface_force = surface_output_dict['surface_force']
    output_vg.surface_moment = surface_output_dict['surface_moment']

    return output_vg