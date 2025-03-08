import numpy as np
import csdl_alpha as csdl

from VortexAD.core.panel_method.steady.higher_order.pre_processor import pre_processor
from VortexAD.core.panel_method.steady.higher_order.mu_sigma_solver import mu_sigma_solver

from VortexAD.core.panel_method.steady.higher_order.compute_vertex_normals import VertexNormals

def linear_doublet_solver(exp_orig_mesh_dict, num_nodes, mesh_mode, rho=1.225):
    print('running pre-processing')
    mesh_dict = pre_processor(exp_orig_mesh_dict, mode=mesh_mode)
    panel_normal = mesh_dict['panel_normal']
    vertices = mesh_dict['points']
    Pc = mesh_dict['panel_center']
    
    IDW = False
    if IDW:

        # IDW to get normal vectors at vertices
        # NOTE: This won't work because of the proximity between upper and lower surface
        # This will cause the z-direction vector to basically cancel out
        # We need to implement a custom operation for this
        
        npa, nv = Pc.shape[1], vertices.shape[1]

        # shape should be nn, nv, np, 3
        D_shape = (num_nodes, nv, npa, 3)
        vertices_expanded = csdl.expand(vertices, D_shape, 'ijk->ijak')
        Pc_expanded = csdl.expand(Pc, D_shape, 'ijk->iajk')

        D = 1/csdl.norm(vertices_expanded-Pc_expanded, axes=(3,))
        D_col_sum = csdl.sum(D, axes=(2,))
        vertex_normal = csdl.Variable(value=np.zeros((num_nodes, nv, 3)))
        for i in csdl.frange(3):
            vertex_normal = vertex_normal.set(
                csdl.slice[0,:,i],
                value=csdl.matvec(D[0,:], panel_normal[0,:,i])/D_col_sum[0,:]
            )

        
        vertex_normal = vertex_normal / csdl.expand(
            csdl.norm(
                vertex_normal,
                axes=(2,)
            ),
            vertex_normal.shape,
            'ij->ija'
        )

        mesh_dict['vertex_normal'] = vertex_normal
    else:
        points2cells = mesh_dict['points2cells']
        vertex_normal_func = VertexNormals(adjacent_panels=points2cells)
        vertex_normal = vertex_normal_func.evaluate(
            vertices=vertices,
            panel_normal=panel_normal
        )
        vertex_normal = vertex_normal / csdl.expand(
            csdl.norm(
                vertex_normal,
                axes=(2,)
            ),
            vertex_normal.shape,
            'ij->ija'
        )
        mesh_dict['vertex_normal'] = vertex_normal


    if False:
        from VortexAD.utils.plot_unstructured import plot_pressure_distribution
        triangles = mesh_dict['cell_point_indices']
        # plot_pressure_distribution(vertices[0,:].value, D[0,0,:].value, connectivity=triangles, interactive=True)
        plot_pressure_distribution(vertices[0,:].value, panel_normal[0,:].value/10., panel_center=Pc[0,:].value, connectivity=triangles, on='cells', interactive=True, top_view=False)
        plot_pressure_distribution(vertices[0,:].value, vertex_normal[0,:].value/10., connectivity=triangles, on='points', interactive=True, top_view=False)
        exit()

    mu, sigma = mu_sigma_solver(num_nodes, mesh_dict, mode=mesh_mode)