import numpy as np 
import csdl_alpha as csdl 
from dataclasses import dataclass
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


def vlm_solver(mesh_list, mesh_velocity_list, alpha_ML=None):
    orig_mesh_dict = {}
    surface_counter = 0

    for i in range(len(mesh_list)):
        surface_name = f'surface_{surface_counter}'
        orig_mesh_dict[surface_name] = {}
        orig_mesh_dict[surface_name]['mesh'] = mesh_list[i]
        orig_mesh_dict[surface_name]['nodal_velocity'] = mesh_velocity_list[i] * -1.
        num_nodes = mesh_list[i].shape[0] # NOTE: CHECK THIS LINE
        surface_counter += 1

    print('running pre-processing')
    mesh_dict = pre_processor(orig_mesh_dict)

    print('solving for circulation strengths')
    gamma = gamma_solver(num_nodes, mesh_dict)

    print('running post-processing')
    surface_output_dict, total_output_dict = post_processor(num_nodes, mesh_dict, gamma, alpha_ML=alpha_ML)

    surface_names = mesh_dict.keys()
    surface_panel_force_points = []
    for surface_name in surface_names:
        surface_panel_force_points.append(mesh_dict[surface_name]['force_eval_points'])


    print('setting up outputs')

    @dataclass
    class Outputs(csdl.VariableGroup):
        total_lift: csdl.Variable # total lift force
        total_drag: csdl.Variable # total drag force
        total_force: csdl.Variable # total force (x,y,z)
        total_moment: csdl.Variable # total moment w.r.t the input reference point
        

        surface_CL: csdl.Variable # CL of each lifting surface
        surface_CDi: csdl.Variable # CDi of each lifting surface
        surface_lift: csdl.Variable # lift of each lifting surface
        surface_drag: csdl.Variable # drag of each lifting surface
        surface_force: csdl.Variable # total force of lifting surface
        surface_moment: csdl.Variable # total moment of lifting surface w.r.t the input reference point

        surface_panel_forces: list # individual panel forces of each lifting surface
        surface_panel_force_points: list
        surface_panel_areas: list
        surface_panel_force_points: list # location of panel forces of each lifting surface (1/4 chord of the mesh panels)
        surface_sectional_cop: list # span-wise sectional center of pressure for each lifting surface
        surface_cop: csdl.Variable # center of pressure for each lifting surface

    
    output_vg = Outputs(
        total_lift = total_output_dict['total_lift'],
        total_drag = total_output_dict['total_drag'],
        total_force = total_output_dict['total_force'],
        total_moment = total_output_dict['total_moment'],

        surface_CL = surface_output_dict['surface_CL'],
        surface_CDi = surface_output_dict['surface_CDi'],
        surface_lift = surface_output_dict['surface_lift'],
        surface_drag = surface_output_dict['surface_drag'],
        surface_force = surface_output_dict['surface_force'],
        surface_moment = surface_output_dict['surface_moment'],
        
        surface_panel_forces = surface_output_dict['surface_panel_forces'],
        surface_panel_areas = [surface['panel_area'] for surface in mesh_dict.values()],
        surface_panel_force_points = surface_panel_force_points,
        surface_sectional_cop = surface_output_dict['surface_sectional_cop'],
        surface_cop = surface_output_dict['surface_cop'],

    )

    

    return output_vg