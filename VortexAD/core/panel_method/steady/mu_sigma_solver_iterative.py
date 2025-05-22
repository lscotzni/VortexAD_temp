import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.panel_method.steady.fixed_wake_representation import fixed_wake_representation
from VortexAD.core.panel_method.steady.compute_source_strength import compute_source_strength

from VortexAD.core.panel_method.source_doublet.source_functions import compute_source_influence_new 
from VortexAD.core.panel_method.source_doublet.doublet_functions import compute_doublet_influence_new
from VortexAD.core.panel_method.vortex_ring.vortex_line_functions import compute_vortex_line_ind_vel


def mu_sigma_solver_iterative(num_nodes, mesh_dict, mode='structured', batch_size=None, bc='Dirichlet', ROM=False):

    if mode == 'structured':
        surface_names = list(mesh_dict.keys())
        num_tot_panels = 0
        for surface in surface_names:
            num_tot_panels += mesh_dict[surface]['num_panels']
    
    elif mode == 'unstructured':
        num_tot_panels = len(mesh_dict['cell_adjacency'])
    
    wake_mesh_dict = fixed_wake_representation(mesh_dict, num_nodes, wake_propagation_dt=100, mesh_mode=mode)
    sigma = compute_source_strength(mesh_dict, num_nodes, num_panels=num_tot_panels, mesh_mode=mode)

    # mesh properties
    coll_point_eval = mesh_dict['panel_center_mod'] # (nn, num_tot_panels, 3)

    coll_point = mesh_dict['panel_center'] # (nn, num_tot_panels, 3)
    panel_corners = mesh_dict['panel_corners'] # (nn, num_tot_panels, 3, 3) 
    panel_x_dir = mesh_dict['panel_x_dir'] # (nn, num_tot_panels, 3)
    panel_y_dir = mesh_dict['panel_y_dir'] # (nn, num_tot_panels, 3)
    panel_normal = mesh_dict['panel_normal'] # (nn, num_tot_panels, 3)
    S_j = mesh_dict['S']
    SL_j = mesh_dict['SL']
    SM_j = mesh_dict['SM']

    # compute RHS via batched matvec product
    batch_size=2
    RHS_batched_func = csdl.experimental.batch_function(compute_aic_mat_vec, batch_size=batch_size, batch_dims=[1]*9+[None, None])

    RHS = RHS_batched_func(
        coll_point_eval.
        coll_point,
        panel_corners,
        panel_x_dir,
        panel_y_dir,
        panel_normal,
        S_j,
        SL_j,
        SM_j,
        sigma,
        'source'
    )

    mu_batched_func = csdl.experimental.batch_function(compute_aic_mat_vec, batch_size=batch_size, batch_dims=[1]*9+[None, None])
    mu_wake_batched_func = csdl.experimental.batch_function(compute_aic_mat_vec, batch_size=batch_size, batch_dims=[1]*9+[None, None])

    def AIC_mu_matvec_gmres(vec):
        def mat_vec_products(v):
            Av_grid_rc = (AIC_batched_func_row(theta_row, theta_col, vec))
        
        Av_matvec = mat_vec_products(v)
        return Av_matvec
    

    from csdl_alpha.src.operations.linalg.linear_solvers.krylov_solver import solve_gmres
    mu = solve_gmres(AIC_mu_matvec_gmres, RHS, transpose_solve=lambda x: solve_gmres(AIC_mu_matvec_gmres, x))

    return mu, sigma, wake_mesh_dict

def compute_aic_mat_vec(coll_point, panel_center, panel_corners, panel_x_dir, panel_y_dir,
                        panel_normal, S_j, SL_j, SM_j, v, mode='doublet'):
    '''
    This function computes the matrix vector product Av, where A is the 
    AIC matrix for the doublets, and v represents the vector that converges
    to the doublet strengths

    The three modes are doublet, source, and wake (corresponding to which AIC matrix to compute)
    '''
    num_nodes = coll_point.shape[0]
    num_eval_pts = coll_point.shape[1]
    num_induced_pts = panel_center.shape[1]
    
    num_interactions = num_eval_pts*num_induced_pts
    expanded_shape = (num_nodes, num_eval_pts, num_induced_pts, 3, 3)
    vectorized_shape = (num_nodes, num_interactions, 3, 3)

    # ============ expanding across columns ============
    coll_point_exp = csdl.expand(coll_point, expanded_shape, 'ijk->ijabk')
    coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

    # ============ expanding across rows ============
    coll_point_j_exp = csdl.expand(coll_point, expanded_shape, 'ijk->iajbk')
    coll_point_j_exp_vec = coll_point_j_exp.reshape(vectorized_shape)

    panel_corners_exp = csdl.expand(panel_corners, expanded_shape, 'ijkl->iajkl')
    panel_corners_exp_vec = panel_corners_exp.reshape(vectorized_shape)

    panel_x_dir_exp = csdl.expand(panel_x_dir, expanded_shape, 'ijk->iajbk')
    panel_x_dir_exp_vec = panel_x_dir_exp.reshape(vectorized_shape)
    panel_y_dir_exp = csdl.expand(panel_y_dir, expanded_shape, 'ijk->iajbk')
    panel_y_dir_exp_vec = panel_y_dir_exp.reshape(vectorized_shape)
    panel_normal_exp = csdl.expand(panel_normal, expanded_shape, 'ijk->iajbk')
    panel_normal_exp_vec = panel_normal_exp.reshape(vectorized_shape)

    S_j_exp = csdl.expand(S_j, expanded_shape[:-1] , 'ijk->iajk')
    S_j_exp_vec = S_j_exp.reshape(vectorized_shape[:-1])

    SL_j_exp = csdl.expand(SL_j, expanded_shape[:-1], 'ijk->iajk')
    SL_j_exp_vec = SL_j_exp.reshape(vectorized_shape[:-1])

    SM_j_exp = csdl.expand(SM_j, expanded_shape[:-1], 'ijk->iajk')
    SM_j_exp_vec = SM_j_exp.reshape(vectorized_shape[:-1])

    a = coll_point_exp_vec - panel_corners_exp_vec # Rc - Ri
    P_JK = coll_point_exp_vec - coll_point_j_exp_vec # RcJ - RcK
    sum_ind = len(a.shape) - 1

    A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
    AL = csdl.sum(a*panel_x_dir_exp_vec, axes=(sum_ind,))
    AM = csdl.sum(a*panel_y_dir_exp_vec, axes=(sum_ind,)) # m-direction projection 
    PN = csdl.sum(P_JK*panel_normal_exp_vec, axes=(sum_ind,)) # normal projection of CP
    print(A.shape)
    B = csdl.Variable(shape=A.shape, value=0.)
    B = B.set(csdl.slice[:,:-1], value=A[:,1:])
    B = B.set(csdl.slice[:,-1], value=A[:,0])

    BL = csdl.Variable(shape=AL.shape, value=0.)
    BL = BL.set(csdl.slice[:,:-1], value=BL[:,1:])
    BL = BL.set(csdl.slice[:,-1], value=BL[:,0])

    BM = csdl.Variable(shape=AM.shape, value=0.)
    BM = BM.set(csdl.slice[:,:-1], value=AM[:,1:])
    BM = BM.set(csdl.slice[:,-1], value=AM[:,0])

    A1 = AM*SL_j_exp_vec - AL*SM_j_exp_vec

    A_list = [A[:,ind] for ind in range(3)]
    AM_list = [AM[:,ind] for ind in range(3)]
    B_list = [B[:,ind] for ind in range(3)]
    BM_list = [BM[:,ind] for ind in range(3)]
    SL_list = [SL_j_exp_vec[:,ind] for ind in range(3)]
    SM_list = [SM_j_exp_vec[:,ind] for ind in range(3)]
    A1_list = [A1[:,ind] for ind in range(3)]
    PN_list = [PN[:,ind] for ind in range(3)]
    S_list = [S_j_exp_vec[:,ind] for ind in range(3)]

    if mode == 'doublet' or mode == 'wake':
        AIC_vec = compute_doublet_influence_new(
            A_list, 
            AM_list, 
            B_list, 
            BM_list, 
            SL_list, 
            SM_list, 
            A1_list, 
            PN_list, 
            mode='potential'
        )
    elif mode == 'source':
        AIC_vec = compute_source_influence_new(
            A_list, 
            AM_list, 
            B_list, 
            BM_list, 
            SL_list, 
            SM_list, 
            A1_list, 
            PN_list, 
            S_list, 
            mode='potential'
        )

    A = AIC_vec.reshape((num_nodes, num_eval_pts, num_induced_pts))
    Av = csdl.einsum(A,v,'ijk,ik->ij')

    return Av

'''
import csdl_alpha as csdl
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

recorder = csdl.Recorder(inline = True)
recorder.start()

n = 6000
batch_size_row = 1000
batch_size_col = 2

# Parameters to construct matrix A
theta_row_val, theta_col_val, vec_val = np.arange(n), np.arange(n), (1.0/(np.arange(n)+1.0)).reshape((n,1))
dummy_input = csdl.Variable(value = np.ones(1,))
theta_row = csdl.Variable(value = theta_row_val)*dummy_input
theta_col = csdl.Variable(value = theta_col_val)*dummy_input 

# B Vector to solve over
b = csdl.Variable(value = vec_val)*dummy_input

# Matvec product function
def matvec(theta_i,theta_j, vec):
    print(f'SHAPES: theta_i: {theta_i.shape}, theta_j: {theta_j.shape}, vec: {vec.shape}')
    n_rows = theta_i.shape[0]
    n_cols = theta_j.shape[0]
    subtraction = (csdl.expand(theta_i, (n_rows, n_cols), action='i->ij') - csdl.expand(theta_j, (n_rows, n_cols), action='i->ji'))+0.1
    A_grid = (1/subtraction/n)**2.0
    assert A_grid.shape == (theta_i.shape[0], theta_j.shape[0])
    return A_grid@vec

# Batch across rows
batched_func_row = csdl.experimental.batch_function(matvec, batch_size_row,[0, None, None])

# Define matvec linear operator
def matvec_gmres(vec):
    Av_grid_rc = (batched_func_row(theta_row, theta_col, vec))
    return Av_grid_rc

# Solve system
from csdl_alpha.src.operations.linalg.linear_solvers.krylov_solver import solve_gmres
x = solve_gmres(matvec_gmres, b, transpose_solve=lambda x: solve_gmres(matvec_gmres, x))

# Outputs and compute derivatives
norm = csdl.norm(x)
sim = csdl.experimental.JaxSimulator(
    recorder,
    additional_inputs=[dummy_input],
    additional_outputs=[norm],
    gpu=True,
    f64=True,
)

import time
start_time = time.time()
sim.compute_totals()
compile_time = time.time() - start_time
print(f'Compile time: {compile_time:.6f} seconds')

num = 3
start_time = time.time()
for i in range(num):
    sim.compute_totals()
average_time = (time.time() - start_time)/num
print(f'Average time for compute_totals: {average_time:.6f} seconds')

'''