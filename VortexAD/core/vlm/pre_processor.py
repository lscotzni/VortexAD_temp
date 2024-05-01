import numpy as np 
import csdl_alpha as csdl

def pre_processor(mesh_dict, alpha):
    mesh_names = mesh_dict.keys()

    for key in mesh_names:
        mesh = csdl.Variable(name=key+'_mesh', value=mesh_dict[key]['mesh'])
        # NOTE: mesh has shape (num_nodes, nc, ns, 3)

        mesh_dict[key]['nc'] = mesh.shape[1]
        mesh_dict[key]['ns'] = mesh.shape[2]

        # bound vortex grid computation
        bound_vortex_mesh = csdl. Variable(shape=mesh.shape, value=0.)
        bound_vortex_mesh = bound_vortex_mesh.set(csdl.slice[:,:-1,:,:], value=(3*mesh[:,:-1,:,:] + mesh[:,1:,:,:])/4)
        bound_vortex_mesh = bound_vortex_mesh.set(csdl.slice[:,-1,:,:],value=mesh[:,-1,:,:] + (mesh[:,-1,:,:] - mesh[:,-2,:,:])/4)

        mesh_dict[key]['bound_vortex_mesh'] = bound_vortex_mesh

        # collocation point computation (center of the vortex rings defined by vortex mesh)
        p1_bd = bound_vortex_mesh[:,:-1, :-1, :]
        p2_bd = bound_vortex_mesh[:,:-1, 1:, :]
        p3_bd = bound_vortex_mesh[:,1:, 1:, :]
        p4_bd = bound_vortex_mesh[:,1:, :-1, :]

        collocation_points = (p1_bd + p2_bd + p3_bd + p4_bd)/4.
        mesh_dict[key]['collocation_points'] = collocation_points

        # panel area and normal vector computation (NOTE: CHECK IF WE NEED TO USE THE MESH OR BOUND VORTEX GRID)
        p1 = mesh[:, :-1, :-1, :]
        p2 = mesh[:, :-1, 1:, :]
        p3 = mesh[:, 1:, 1:, :]
        p4 = mesh[:, 1:, :-1, :]

        # panel diagonal vectors
        A = p3_bd - p1_bd
        B = p2_bd - p4_bd
        normal_dir = csdl.cross(A, B, axis=3)
        panel_area = csdl.norm(normal_dir, axes=(3,)) / 2.

        # vector normalization
        normal_vec = normal_dir/(csdl.expand(panel_area, out_shape=normal_dir.shape, action='ijk->ijka') + 1.e-12)

        mesh_dict[key]['panel_area'] = panel_area
        mesh_dict[key]['bd_normal_vec'] = normal_vec

        wetted_area = csdl.sum(panel_area, axes=(1,2))
        mesh_dict[key]['wetted_area'] = wetted_area

        # NOTE: ADD BOUND VECTOR COMPUTATION FOR WHERE THE FORCE IS APPLIED
        # THIS IS AT THE LEADING EDGE OF THE BOUND VORTEX GRID

    return mesh_dict