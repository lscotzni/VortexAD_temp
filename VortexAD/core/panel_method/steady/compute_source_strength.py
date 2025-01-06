import numpy as np
import csdl_alpha as csdl 

def compute_source_strength(mesh_dict, num_nodes, num_panels, mesh_mode='structured'):
    if mesh_mode == 'structured':
        surface_names = list(mesh_dict.keys())

        sigma = csdl.Variable(shape=(num_nodes, num_panels), value=0.)
        start, stop = 0, 0
        for surface in surface_names:
            num_surf_panels = mesh_dict[surface]['num_panels']
            stop += num_surf_panels

            center_pt_velocity = mesh_dict[surface]['nodal_cp_velocity']
            coll_point_vel = mesh_dict[surface]['coll_point_velocity']
            panel_normal = mesh_dict[surface]['panel_normal']

            if coll_point_vel:
                total_vel = center_pt_velocity+coll_point_vel
            else:
                total_vel = center_pt_velocity

            # vel_projection = csdl.einsum(coll_point_velocity, panel_normal, action='ijklm,ijklm->ijkl')
            vel_projection = csdl.sum(-total_vel*panel_normal, axes=(3,))

            sigma = sigma.set(csdl.slice[:,start:stop], value=csdl.reshape(vel_projection, shape=(num_nodes, num_surf_panels)))
            start += num_surf_panels

    elif mesh_mode == 'unstructured':
        coll_point_velocity = mesh_dict['coll_point_velocity']
        panel_normal = mesh_dict['panel_normal']
        sigma = -csdl.sum(
            coll_point_velocity*panel_normal,
            axes=(2,)
        )

    return sigma # VECTORIZED in shape=(num_nodes, nt, num_surf_panels)