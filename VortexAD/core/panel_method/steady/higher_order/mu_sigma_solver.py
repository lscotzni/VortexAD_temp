import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.panel_method.steady.fixed_wake_representation import fixed_wake_representation
from VortexAD.core.panel_method.steady.compute_source_strength import compute_source_strength

from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_influence_new 

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
        # AIC_mu, AIC_sigma = unstructured_AIC_computation_old(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels)
        AIC_mu, AIC_sigma = unstructured_AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels)

    
def AIC_computation():
    pass

def unstructured_AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels):
    '''
    Method to compute the higher-order AIC matrices (both mu and sigma)
    NOTE: The process to generate the AIC sigma matrix is mostly unchanged
    - constant strength panels, with control points now at the mesh vertices
    '''
    vertices = mesh_dict['points']
    num_vertices = vertices.shape[1]
    vertex_normal = mesh_dict['vertex_normal']
    k = 1.e-6
    control_points = vertices - vertex_normal*k

    panel_normal = mesh_dict['panel_normal']
    panel_center = mesh_dict['panel_center']
    num_panels = panel_center.shape[1]
    '''
    Shape of AIC mu: (num_nodes, num_vertices, num_vertices)
    Shape of AIC sigma: (num_nodes, num_vertices, num_panels)
    '''
    AIC_mu = csdl.Variable(value=np.zeros((num_nodes, num_vertices, num_vertices)))

    # ======================== COMPUTING THE AIC MATRIX FOR SOURCE STRENGTHS SIGMA ========================
    eval_pt = vertices # evaluation point
    eval_normal = vertex_normal # normal vector at evaluation point

    panel_corners_j = mesh_dict['panel_corners']
    coll_point_j = mesh_dict['panel_center']
    panel_x_dir_j = mesh_dict['panel_x_dir']
    panel_y_dir_j = mesh_dict['panel_y_dir']
    panel_normal_j = mesh_dict['panel_normal']

    S_j = mesh_dict['S']
    SL_j = mesh_dict['SL']
    SM_j = mesh_dict['SM']

    num_interactions = num_vertices*num_panels
    expanded_shape = (num_nodes, num_vertices, num_panels, 3, 3)
    vectorized_shape = (num_nodes, num_interactions, 3, 3)

    eval_pt_exp = csdl.expand(eval_pt, expanded_shape, 'jkl->jkabl')
    eval_pt_exp_vec = eval_pt_exp.reshape(vectorized_shape)

    eval_normal_grid = csdl.expand(eval_normal, (num_nodes, num_vertices, num_panels, 3), 'jkl->jkal')
    # eval_normal_grid = eval_normal_exp.reshape((num_nodes, num_vertices, num_panels, 3)) # for normal projection

    panel_corners_j_exp = csdl.expand(panel_corners_j, expanded_shape, 'jklm->jaklm')
    panel_corners_j_exp_vec = panel_corners_j_exp.reshape(vectorized_shape)

    coll_point_j_exp = csdl.expand(coll_point_j, expanded_shape, 'jkl->jakbl')
    coll_point_j_exp_vec = coll_point_j_exp.reshape(vectorized_shape)

    panel_x_dir_j_exp = csdl.expand(panel_x_dir_j, expanded_shape, 'jkl->jakbl')
    panel_x_dir_j_exp_vec = panel_x_dir_j_exp.reshape(vectorized_shape)
    panel_y_dir_j_exp = csdl.expand(panel_y_dir_j, expanded_shape, 'jkl->jakbl')
    panel_y_dir_j_exp_vec = panel_y_dir_j_exp.reshape(vectorized_shape)
    panel_normal_j_exp = csdl.expand(panel_normal_j, expanded_shape, 'jkl->jakbl')
    panel_normal_j_exp_vec = panel_normal_j_exp.reshape(vectorized_shape)

    S_j_exp = csdl.expand(S_j, expanded_shape[:-1], 'jkl->jakl')
    S_j_exp_vec = S_j_exp.reshape(vectorized_shape[:-1])

    SL_j_exp = csdl.expand(SL_j, expanded_shape[:-1], 'jkl->jakl')
    SL_j_exp_vec = SL_j_exp.reshape(vectorized_shape[:-1])

    SM_j_exp = csdl.expand(SM_j, expanded_shape[:-1], 'jkl->jakl')
    SM_j_exp_vec = SM_j_exp.reshape(vectorized_shape[:-1])
    
    a = eval_pt_exp_vec - panel_corners_j_exp_vec # Rc - Ri
    P_JK = eval_pt_exp_vec - coll_point_j_exp_vec # RcJ - RcK
    sum_ind = len(a.shape) - 1

    A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
    AL = csdl.sum(a*panel_x_dir_j_exp_vec, axes=(sum_ind,))
    AM = csdl.sum(a*panel_y_dir_j_exp_vec, axes=(sum_ind,)) # m-direction projection 
    PN = csdl.sum(P_JK*panel_normal_j_exp_vec, axes=(sum_ind,)) # normal projection of CP

    B = csdl.Variable(shape=A.shape, value=0.)
    B = B.set(csdl.slice[:,:,:-1], value=A[:,:,1:])
    B = B.set(csdl.slice[:,:,-1], value=A[:,:,0])

    BL = csdl.Variable(shape=AL.shape, value=0.)
    BL = BL.set(csdl.slice[:,:,:-1], value=BL[:,:,1:])
    BL = BL.set(csdl.slice[:,:,-1], value=BL[:,:,0])

    BM = csdl.Variable(shape=AM.shape, value=0.)
    BM = BM.set(csdl.slice[:,:,:-1], value=AM[:,:,1:])
    BM = BM.set(csdl.slice[:,:,-1], value=AM[:,:,0])

    A1 = AM*SL_j_exp_vec - AL*SM_j_exp_vec

    A = A.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
    AM = AM.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
    B = B.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
    BM = BM.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
    SL_j_exp_vec = SL_j_exp_vec.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
    SM_j_exp_vec = SM_j_exp_vec.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
    A1 = A1.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
    PN = PN.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
    S_j_exp_vec = S_j_exp_vec.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')

    A_list = [A[:,:,ind] for ind in range(3)]
    AM_list = [AM[:,:,ind] for ind in range(3)]
    B_list = [B[:,:,ind] for ind in range(3)]
    BM_list = [BM[:,:,ind] for ind in range(3)]
    SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(3)]
    SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(3)]
    A1_list = [A1[:,:,ind] for ind in range(3)]
    PN_list = [PN[:,:,ind] for ind in range(3)]
    S_list = [S_j_exp_vec[:,:,ind] for ind in range(3)]

    source_vel_vec = compute_source_influence_new(
        A_list, 
        AM_list, 
        B_list, 
        BM_list, 
        SL_list, 
        SM_list, 
        A1_list, 
        PN_list, 
        S_list, 
        panel_x_dir_j_exp_vec[:,:,0,:],
        panel_y_dir_j_exp_vec[:,:,0,:],
        panel_normal_j_exp_vec[:,:,0,:],
        mode='velocity'
    )
    source_vel = source_vel_vec.reshape((num_nodes, num_vertices, num_panels, 3))
    source_vel_normal_proj = csdl.sum(source_vel*eval_normal_grid, axes=(3,))

    AIC_sigma = source_vel_normal_proj
    
    # ======================== COMPUTING THE AIC MATRIX FOR DOUBLET STRENGTHS MU ========================
    # shape is (num_nodes, num_vertices, num_vertices)
    panel_vertex_deltas = mesh_dict['panel_vertex_deltas']
    cell_point_indices = mesh_dict['cell_point_indices']
    panel_to_vertex_tmat = csdl.Variable(
        value=np.ones((num_nodes, num_panels, 3, 3))
    )
    for i in csdl.frange(3):
        panel_to_vertex_tmat = panel_to_vertex_tmat.set(
            csdl.slice[:,:,i,1:],
            value=panel_vertex_deltas[:,:,i,:2]
        )
    
    # panel_to_vertex_tmat = panel_to_vertex_tmat.set(
    #     csdl.slice[:,:,0,1:],
    #     value=panel_vertex_deltas[:,:,list(cell_point_indices[:,0]),:2]
    # )
    # panel_to_vertex_tmat = panel_to_vertex_tmat.set(
    #     csdl.slice[:,:,1,1:],
    #     value=panel_vertex_deltas[:,:,list(cell_point_indices[:,1]),:2]
    # )
    # panel_to_vertex_tmat = panel_to_vertex_tmat.set(
    #     csdl.slice[:,:,2,1:],
    #     value=panel_vertex_deltas[:,:,list(cell_point_indices[:,2]),:2]
    # )

    inv_transformation = matrix_inverse_computation(panel_to_vertex_tmat)

    # populating matrix
    global_vertex_to_panel_tmat = csdl.Variable(value=np.zeros((num_nodes, 3*num_panels, num_vertices)))
    # global_vertex_to_panel_tmat = csdl.Variable(value=np.zeros((num_nodes, num_panels, num_vertices, 3)))
    a_list = cell_point_indices[:,0].tolist()
    b_list = cell_point_indices[:,1].tolist()
    c_list = cell_point_indices[:,2].tolist()
    num_panel_list = [i for i in range(num_panels)]

    for i, a, b, c in csdl.frange(vals=(num_panel_list, a_list, b_list, c_list), inline_lazy_stack=True):
        global_vertex_to_panel_tmat = global_vertex_to_panel_tmat.set(
            csdl.slice[:,3*i,[a,b,c]],
            value=inv_transformation[:,i,0,:]
        )
        global_vertex_to_panel_tmat = global_vertex_to_panel_tmat.set(
            csdl.slice[:,3*i+1,[a,b,c]],
            value=inv_transformation[:,i,1,:]
        )
        global_vertex_to_panel_tmat = global_vertex_to_panel_tmat.set(
            csdl.slice[:,3*i+2,[a,b,c]],
            value=inv_transformation[:,i,2,:]
        )

    # for i, a, b, c in csdl.frange(vals=(num_panel_list, a_list, b_list, c_list), inline_lazy_stack=True):
    #     global_vertex_to_panel_tmat = global_vertex_to_panel_tmat.set(
    #         csdl.slice[:,i,[a,b,c],0],
    #         value=inv_transformation[:,i,0,:]
    #     )
    #     global_vertex_to_panel_tmat = global_vertex_to_panel_tmat.set(
    #         csdl.slice[:,i,[a,b,c],1],
    #         value=inv_transformation[:,i,1,:]
    #     )
    #     global_vertex_to_panel_tmat = global_vertex_to_panel_tmat.set(
    #         csdl.slice[:,i,[a,b,c],2],
    #         value=inv_transformation[:,i,2,:]
    #     )
    
    # TODO: COMPUTE THE HIGHER-ORDER INTERACTIONS NOW
        
    edge_vec = mesh_dict['edge_vec']
    edge_normal = mesh_dict['edge_normal']
    panel_center_j = mesh_dict['panel_center']
    panel_corners_j = panel_corners_j # panel corners

    expanded_shape = (num_nodes, num_vertices, num_panels, 3, 3)
    vectorized_shape = (num_nodes, num_interactions, 3, 3)
    # eval_pt_exp = csdl.expand(eval_pt, expanded_shape, 'ijk->ijabk')
    local_vertex_pos = mesh_dict['local_vertex_position'] # nn, np, nv, 3
    local_vertex_pos_exp = local_vertex_pos.reshape(vectorized_shape[:-1])
    '''
    variables to use from above:
    - eval_pt_exp_vec
    - panel_corners_j_exp_vec
    '''

    edge_normal_exp = csdl.expand(edge_normal, expanded_shape, 'ijkl->iajkl')
    edge_normal_exp_vec = edge_normal_exp.reshape(vectorized_shape)

    edge_vec_exp = csdl.expand(edge_vec, expanded_shape, 'ijkl->iajkl')
    edge_vec_exp_vec = edge_vec_exp.reshape(vectorized_shape)

    # a_bar = (vertices - panel_corners_j) * edge_normal

    # h = (vertices - panel_center_j) * panel_normal
    # l1 = (vertices - panel_corners_j) * edge_vec[0] # need to adjust sizes for the 3 panel corners
    # l2 = (vertices - panel_corners_j) * edge_vec[1] # need to adjust sizes for the 3 panel corners

    # g = (a_bar + h)**0.5
    # s1 = (l1**2 + g**2)**0.5
    # s2 = (l2**2 + g**2)**0.5
    # c1 = g**2 + abs(h)*s1
    # c2 = g**2 + abs(h)*s2

    eval_pt_panel_corner_delta = eval_pt_exp_vec - panel_corners_j_exp_vec

    a_bar = csdl.sum(
        (eval_pt_exp_vec - panel_corners_j_exp_vec) * edge_normal_exp_vec,
        axes=(3,),
    )
    h = edge_normal_exp_vec[:,:,2] # already in rotated frame

    l1 = csdl.sum(
        eval_pt_panel_corner_delta * edge_vec_exp_vec,
        axes=(3,)
    ) 
    l2 = csdl.sum(
        eval_pt_panel_corner_delta * edge_vec_exp_vec,
        axes=(3,)
    ) 
    # NOTE: l1, l2 NEED TO BE FIXED

    g = (a_bar + h)**0.5
    s1 = (l1**2 + g**2)**0.5
    s2 = (l2**2 + g**2)**0.5
    c1 = g**2 + (h**2+1.e-12)**0.5*s1 # CHECK THE NUMERICAL SOFTENING HERE
    c2 = g**2 + (h**2+1.e-12)**0.5*s2 # CHECK THE NUMERICAL SOFTENING HERE
    
    return AIC_mu, AIC_sigma

def unstructured_AIC_computation_old(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels):
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

def matrix_inverse_computation(transformation_matrix):
    m = transformation_matrix
    xi1 = m[:,:,0,1]
    xi2 = m[:,:,1,1]
    xi3 = m[:,:,2,1]
    eta1 = m[:,:,0,2]
    eta2 = m[:,:,1,2]
    eta3 = m[:,:,2,2]
    det_m = xi2*eta3 - eta2*xi3 - \
        xi1*(eta3-eta2) + eta1*(xi3-xi2)
    det_m_exp = csdl.expand(det_m, m.shape, 'ij->ijab')
    
    adj_m = csdl.Variable(value=np.zeros(m.shape))
    adj_m = adj_m.set(csdl.slice[:,:,0,0], value = xi2*eta3 - xi3*eta2)
    adj_m = adj_m.set(csdl.slice[:,:,1,0], value = -(eta3-eta2))
    adj_m = adj_m.set(csdl.slice[:,:,2,0], value = (xi3-xi2))
    adj_m = adj_m.set(csdl.slice[:,:,0,1], value = -(xi1*eta3 - xi3*eta1))
    adj_m = adj_m.set(csdl.slice[:,:,1,1], value = eta3 - eta1)
    adj_m = adj_m.set(csdl.slice[:,:,2,1], value = -(xi3 - xi1))
    adj_m = adj_m.set(csdl.slice[:,:,0,2], value = xi1**eta2 - xi2*eta1)
    adj_m = adj_m.set(csdl.slice[:,:,1,2], value = -(eta2 - eta1))
    adj_m = adj_m.set(csdl.slice[:,:,2,2], value = xi2 - xi1)
    
    inv_m = adj_m/det_m_exp
    return inv_m