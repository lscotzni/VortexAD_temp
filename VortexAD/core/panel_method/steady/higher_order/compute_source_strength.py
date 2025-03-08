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

            nodal_velocity = mesh_dict[surface]['nodal_velocity']
            coll_point_vel = mesh_dict[surface]['coll_point_velocity']
            # NOTE: the above line is the actuation velocity at the collocation points
            # we would need this value at the nodes if there is actuation
            vertex_normal = mesh_dict[surface]['vertex_normal']

            if coll_point_vel:
                total_vel = nodal_velocity+coll_point_vel
            else:
                total_vel = nodal_velocity

            # vel_projection = csdl.einsum(coll_point_velocity, vertex_normal, action='ijklm,ijklm->ijkl')
            vel_projection = csdl.sum(-total_vel*vertex_normal, axes=(3,))

            sigma = sigma.set(csdl.slice[:,start:stop], value=csdl.reshape(vel_projection, shape=(num_nodes, num_surf_panels)))
            start += num_surf_panels

    elif mesh_mode == 'unstructured':
        nodal_velocity = mesh_dict['nodal_velocity']
        vertex_normal = mesh_dict['vertex_normal']
        sigma = -csdl.sum(
            nodal_velocity*vertex_normal,
            axes=(2,)
        )

    return sigma # VECTORIZED in shape=(num_nodes, nt, num_surf_panels)