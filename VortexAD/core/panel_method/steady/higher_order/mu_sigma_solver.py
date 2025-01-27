import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.panel_method.steady.fixed_wake_representation import fixed_wake_representation
from VortexAD.core.panel_method.steady.compute_source_strength import compute_source_strength

from VortexAD.core.panel_method.steady.higher_order.recursions import H_113, F_111, F_113, F_123, F_213

def mu_sigma_solver(num_nodes, mesh_dict, mode='structured'):

    if mode == 'structured':
        surface_names = list(mesh_dict.keys())
        num_tot_panels = 0
        for surface in surface_names:
            num_tot_panels += mesh_dict[surface]['num_panels']
    
    elif mode == 'unstructured':
        num_tot_panels = len(mesh_dict['cell_adjacency'])

    # wake_mesh_dict = fixed_wake_representation(mesh_dict, num_nodes, wake_propagation_dt=0.001)
    wake_mesh_dict = fixed_wake_representation(mesh_dict, num_nodes, wake_propagation_dt=10, mesh_mode=mode)

    sigma = compute_source_strength(mesh_dict, num_nodes, num_panels=num_tot_panels, mesh_mode=mode)
    # NOTE: 
    # sigma is computed at the panel centers (for constant-strength)
    # the AIC matrix will compute induced velocities at the mesh vertices 
    # this will make the AIC matrix non-square of shape (num_vertices, num_panels)

    # static AIC matrices for linear system solve
    if mode == 'structured':
        AIC_mu, AIC_sigma = AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels, surface_names)
    elif mode == 'unstructured':
        AIC_mu, AIC_sigma = unstructured_AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels)

    
def AIC_computation():
    pass

def unstructured_AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels):
    vertices = mesh_dict['points']
    num_vertices = vertices.shape[1]
    vertex_normal = mesh_dict['vertex_normal']
    k = 1.e-5
    control_points = vertices - vertex_normal*k

    panel_normal = mesh_dict['panel_normal']
    panel_center = mesh_dict['panel_center']
    num_panels = panel_center.shape[1]

    mu_AIC_shape = (num_nodes, num_vertices, num_vertices, 3)
    sigma_AIC_shape = (num_nodes, num_vertices, num_panels, 3)

    # panel 1
    for i in csdl.frange(1):
        v1 = vertices[0,0,:]
        v2 = vertices[0,1,:]
        v1_normal = vertex_normal[0,0,:]
        panel_center = mesh_dict['panel_center'][0,0,:]

        control_point = v1 - v1_normal*k

        edge_normal = mesh_dict['edge_normal'][0,0,:,:] # panel 0, 3 edges, 3 components
        edge_vec = mesh_dict['edge_vec'][0,0,:,:]
        panel_x_dir = mesh_dict['panel_x_dir'][0,0,:]
        panel_y_dir = mesh_dict['panel_y_dir'][0,0,:]
        panel_normal = mesh_dict['panel_normal'][0,0,:]

        dp = control_point - panel_center
        x = csdl.sum(dp*panel_x_dir)
        y = csdl.sum(dp*panel_y_dir)
        h = csdl.sum(dp*panel_normal)

        a_bar = csdl.sum((control_point-v1)*edge_normal[0,:])

        g = (a_bar**2 + h**2)**0.5

        l1 = csdl.sum(edge_vec[0,:]*(control_point-v1))
        l2 = csdl.sum(edge_vec[0,:]*(control_point-v2))

        s1 = (l1**2 + g**2)**0.5
        s2 = (l2**2 + g**2)**0.5

        c1 = g**2 + h*s1
        c2 = g**2 + h*s2

        H113 = H_113(3*[a_bar], 3*[l1], 3*[l2], 3*[c1], 3*[c2], h)
        F111 = F_111(l1, l2, g)
        nu_xi = edge_normal[0]
        nu_eta = edge_normal[1]
        R1 = csdl.norm(control_point - v1)
        R2 = csdl.norm(control_point - v2)
        xi1 = csdl.sum((v1 - panel_center)*panel_x_dir)
        eta1 = csdl.sum((v1 - panel_center)*panel_y_dir)
        xi2 = csdl.sum((v2 - panel_center)*panel_x_dir)
        eta2 = csdl.sum((v2 - panel_center)*panel_y_dir)
        point = control_point-panel_center
        xi = [xi1, xi2]
        eta = [eta1, eta2]

        F113 = F_113(g, nu_eta, nu_xi, R1, R2, xi, eta, point)
        F123 = F_123(a_bar, nu_xi, F113, nu_eta, R1, R2)
        F213 = F_213(a_bar, nu_xi, F113, nu_eta, R1, R2)

        sub_AIC = csdl.Variable(value=np.zeros((num_nodes, 3,3)))
        sub_AIC = sub_AIC.set(
            csdl.slice[:,0,0],
            value = -h*csdl.sum(edge_normal[:,0]*F113)
        )
        sub_AIC = sub_AIC.set(
            csdl.slice[:,0,1],
            value = -h*csdl.sum(edge_normal[:,0]*(F213+x*F113)) + h*H113
        )
        sub_AIC = sub_AIC.set(
            csdl.slice[:,0,2],
            value = -h*csdl.sum(edge_normal[:,0]*(F123 + y*F113))
        )
        sub_AIC = sub_AIC.set(
            csdl.slice[:,1,0],
            value = -h*csdl.sum(edge_normal[:,1]*F113)
        )
        sub_AIC = sub_AIC.set(
            csdl.slice[:,1,1],
            value = -h*csdl.sum(edge_normal[:,1]*(F213 + x*F113))
        )
        sub_AIC = sub_AIC.set(
            csdl.slice[:,1,2],
            value = -h*csdl.sum(edge_normal[:,1]*(F123 + y*F113)) + h*H113
        )
        sub_AIC = sub_AIC.set(
            csdl.slice[:,2,0],
            value = -csdl.sum(a_bar*F113)
        )
        sub_AIC = sub_AIC.set(
            csdl.slice[:,2,1],
            value = -csdl.sum(a_bar*(F213 + x*F113) - edge_normal[:,0]*F111)
        )
        sub_AIC = sub_AIC.set(
            csdl.slice[:,2,2],
            value = -csdl.sum(a_bar*(F123 + y*F113) - edge_normal[:,1]*F111)
        )

        asdf = csdl.matvec(sub_AIC[0,:].T(), v1_normal)
        graph = csdl.get_current_recorder().active_graph
    graph.visualize('subgraph_ho')
    print(len(graph.node_table))
    exit()

