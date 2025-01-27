import numpy as np
import csdl_alpha as csdl

from VortexAD.core.panel_method.steady.higher_order.pre_processor import pre_processor
from VortexAD.core.panel_method.steady.higher_order.mu_sigma_solver import mu_sigma_solver

def linear_doublet_solver(exp_orig_mesh_dict, num_nodes, mesh_mode):
    print('running pre-processing')
    mesh_dict = pre_processor(exp_orig_mesh_dict, mode=mesh_mode)

    # IDW to get normal vectors at vertices
    panel_normal = mesh_dict['panel_normal']
    vertices = mesh_dict['points']
    Pc = mesh_dict['panel_center']
    npa, nv = Pc.shape[1], vertices.shape[1]

    # shape should be nn, nv, np, 3
    D_shape = (num_nodes, nv, npa, 3)
    vertices_expanded = csdl.expand(vertices, D_shape, 'ijk->ijak')
    Pc_expanded = csdl.expand(Pc, D_shape, 'ijk->iajk')

    D = 1/csdl.norm(vertices_expanded-Pc_expanded, axes=(3,))

    vertex_normal = csdl.Variable(value=np.zeros((num_nodes, nv, 3)))
    for i in csdl.frange(3):
        vertex_normal = vertex_normal.set(
            csdl.slice[0,:,i],
            value=csdl.matvec(D[0,:], panel_normal[0,:,i])
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

    mu, sigma = mu_sigma_solver(num_nodes, mesh_dict, mode=mesh_mode)