import numpy as np
import csdl_alpha as csdl 

import time

def perturbation_velocity_FD_K_P(mu_grid, panel_center_dl, panel_center_dm):
    ql = csdl.Variable(shape=mu_grid.shape, value=0.)
    qm = csdl.Variable(shape=mu_grid.shape, value=0.)

    ql = ql.set(csdl.slice[:,:,1:-1,:], value=-(mu_grid[:,:,2:,:] - mu_grid[:,:,:-2,:])/2./((panel_center_dl[:,:,1:-1,:,0]+panel_center_dl[:,:,1:-1,:,1])/2))
    ql = ql.set(csdl.slice[:,:,0,:], value=-(-3*mu_grid[:,:,0,:]+4*mu_grid[:,:,1,:]-mu_grid[:,:,2,:])/2./((panel_center_dl[:,:,0,:,0]+panel_center_dl[:,:,1,:,0])/2))
    ql = ql.set(csdl.slice[:,:,-1,:], value=-(3*mu_grid[:,:,-1,:]-4*mu_grid[:,:,-2,:]+mu_grid[:,:,-3,:])/2./((panel_center_dl[:,:,-1,:,1]+panel_center_dl[:,:,-2,:,1])/2))

    qm = qm.set(csdl.slice[:,:,:,1:-1], value=-(mu_grid[:,:,:,2:] - mu_grid[:,:,:,:-2])/2./((panel_center_dm[:,:,:,1:-1,0]+panel_center_dm[:,:,:,1:-1,1])/2))
    qm = qm.set(csdl.slice[:,:,:,0], value=-(-3*mu_grid[:,:,:,0]+4*mu_grid[:,:,:,1]-mu_grid[:,:,:,2])/2./((panel_center_dm[:,:,:,0,0]+panel_center_dm[:,:,:,1,0])/2))
    qm = qm.set(csdl.slice[:,:,:,-1], value=-(3*mu_grid[:,:,:,-1]-4*mu_grid[:,:,:,-2]+mu_grid[:,:,:,-3])/2./((panel_center_dm[:,:,:,-1,1]+panel_center_dm[:,:,:,-2,1])/2))

    return ql, qm
            

def perturbation_velocity_FD(mu_grid, dl, dm):
    ql, qm = csdl.Variable(shape=mu_grid.shape, value=0.), csdl.Variable(shape=mu_grid.shape, value=0.)
    ql = ql.set(csdl.slice[:,:,1:-1,:], value=(mu_grid[:,:,2:,:] - mu_grid[:,:,0:-2,:]) / (dl[:,:,1:-1,:,0] + dl[:,:,1:-1,:,1])) # all panels except TE
    # ql = ql.set(csdl.slice[:,:,0,:], value=(mu_grid[:,:,1,:] - mu_grid[:,:,-1,:]) / (dl[:,:,0,:,0] + dl[:,:,0,:,1])) # TE on lower surface
    ql = ql.set(csdl.slice[:,:,0,:], value=(-3*mu_grid[:,:,0,:] + 4*mu_grid[:,:,1,:] - mu_grid[:,:,2,:]) / (3*dl[:,:,1,:,0] - dl[:,:,1,:,1])) # TE on lower surface
    # ql = ql.set(csdl.slice[:,:,-1,:], value=(mu_grid[:,:,0,:] - mu_grid[:,:,-2,:]) / (dl[:,:,-1,:,0] + dl[:,:,-1,:,1])) # TE on upper surface
    ql = ql.set(csdl.slice[:,:,-1,:], value=(3*mu_grid[:,:,-1,:] - 4*mu_grid[:,:,-2,:] + mu_grid[:,:,-3,:]) / (3*dl[:,:,-2,:,1] - dl[:,:,-2,:,0])) # TE on upper surface

    qm = qm.set(csdl.slice[:,:,:,1:-1], value=(mu_grid[:,:,:,2:] - mu_grid[:,:,:,0:-2]) / (dm[:,:,:,1:-1,0] + dm[:,:,:,1:-1,1])) # all panels expect wing tips
    qm = qm.set(csdl.slice[:,:,:,0], value=(-3*mu_grid[:,:,:,0] + 4*mu_grid[:,:,:,1] - mu_grid[:,:,:,2]) / (3*dm[:,:,:,1,0] - dm[:,:,:,1,1]))
    qm = qm.set(csdl.slice[:,:,:,-1], value=(3*mu_grid[:,:,:,-1] - 4*mu_grid[:,:,:,-2] + mu_grid[:,:,:,-3]) / (3*dm[:,:,:,-2,1] - dm[:,:,:,-2,0]))

    return -ql, -qm

def least_squares_velocity_old(mu_grid, delta_coll_point):
    num_nodes, nt = mu_grid.shape[0], mu_grid.shape[1]
    grid_shape = mu_grid.shape[2:]
    nc_panels, ns_panels = grid_shape[0], grid_shape[1]
    num_panels = nc_panels*ns_panels
    num_deltas = 4*(ns_panels-2)*(nc_panels-2) + 3*(2)*(ns_panels-2+nc_panels-2) + 2*4
    num_derivatives = num_panels*2
    # the above count is done first through central panels, then non-corner edge panels, then corner panels
    delta_system_matrix = csdl.Variable(shape=(num_nodes, nt, num_deltas, num_derivatives), value=0.)
    delta_mu_vec = csdl.Variable(shape=(num_nodes, nt, num_deltas), value=0.)
    TE_deltas = [2] + [3]*(ns_panels-2) + [2]
    center_deltas = [3] + [4]*(ns_panels-2) + [3]
    deltas_per_panel = TE_deltas + center_deltas*(nc_panels-2) + TE_deltas
    
    deltas_per_panel_grid = np.array(deltas_per_panel).reshape((nc_panels, ns_panels))

    mu_grid_vec = mu_grid.reshape((num_nodes, nt, num_panels))
    start, stop = 0, 0
    panel_ind = 0
    for i in range(nc_panels):
        print('i:', i)
        for j in range(ns_panels):
            print('j:',j)
            num_d_panel = deltas_per_panel_grid[i,j]
            stop += num_d_panel
            deltas_sub_matrix = csdl.Variable(shape=(num_nodes, nt, num_d_panel, 2), value=0.)
            sub_delta_mu_vec = csdl.Variable(shape=(num_nodes, nt, num_d_panel), value=0.)

            mu_panel = mu_grid[:,:,i,j]
            mu_panel_exp = csdl.expand(mu_panel, sub_delta_mu_vec.shape, 'ij->ija')
            mu_adjacent = csdl.Variable(shape=mu_panel_exp.shape, value=0.)

            if i == 0: # first chordwise panel
                mu_adjacent = mu_adjacent.set(csdl.slice[:,:,0], value=mu_grid[:,:,i+1,j])
                deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,0,:], value=delta_coll_point[:,:,i,j,1,:])
                if j == 0: # 2 panels (first spanwise)
                    mu_adjacent = mu_adjacent.set(csdl.slice[:,:,1], value=mu_grid[:,:,i,j+1])
                    deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,1,:], value=delta_coll_point[:,:,i,j,3,:])
                elif j == ns_panels-1: # 2 panels  (last spanwise)
                    mu_adjacent = mu_adjacent.set(csdl.slice[:,:,1], value=mu_grid[:,:,i,j-1])
                    deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,1,:], value=delta_coll_point[:,:,i,j,2,:])
                else: # 3 panels
                    mu_adjacent = mu_adjacent.set(csdl.slice[:,:,1], value=mu_grid[:,:,i,j-1])
                    mu_adjacent = mu_adjacent.set(csdl.slice[:,:,2], value=mu_grid[:,:,i,j+1])
                    deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,1,:], value=delta_coll_point[:,:,i,j,2,:])
                    deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,2,:], value=delta_coll_point[:,:,i,j,3,:])

            elif i == nc_panels-1: # last chordwise panel
                mu_adjacent = mu_adjacent.set(csdl.slice[:,:,0], value=mu_grid[:,:,i-1,j])
                deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,0,:], value=delta_coll_point[:,:,i,j,0,:])
                if j == 0: # 2 panels (first spanwise)
                    mu_adjacent = mu_adjacent.set(csdl.slice[:,:,1], value=mu_grid[:,:,i,j+1])
                    deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,1,:], value=delta_coll_point[:,:,i,j,3,:])
                elif j == ns_panels-1: # 2 panels (last spanwise)
                    mu_adjacent = mu_adjacent.set(csdl.slice[:,:,1], value=mu_grid[:,:,i,j-1])
                    deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,1,:], value=delta_coll_point[:,:,i,j,2,:])
                else: # 3 panels
                    mu_adjacent = mu_adjacent.set(csdl.slice[:,:,1], value=mu_grid[:,:,i,j-1])
                    mu_adjacent = mu_adjacent.set(csdl.slice[:,:,2], value=mu_grid[:,:,i,j+1])
                    deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,1,:], value=delta_coll_point[:,:,i,j,2,:])
                    deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,2,:], value=delta_coll_point[:,:,i,j,3,:])

            else: # 4 panels
                mu_adjacent = mu_adjacent.set(csdl.slice[:,:,0], value=mu_grid[:,:,i-1,j])
                mu_adjacent = mu_adjacent.set(csdl.slice[:,:,1], value=mu_grid[:,:,i+1,j])
                deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,0,:], value=delta_coll_point[:,:,i,j,0,:])
                deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,1,:], value=delta_coll_point[:,:,i,j,1,:])
                if j == 0:
                    mu_adjacent = mu_adjacent.set(csdl.slice[:,:,2], value=mu_grid[:,:,i,j+1])
                    deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,2,:], value=delta_coll_point[:,:,i,j,3,:])
                elif j == ns_panels-1:
                    mu_adjacent = mu_adjacent.set(csdl.slice[:,:,2], value=mu_grid[:,:,i,j-1])
                    deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,2,:], value=delta_coll_point[:,:,i,j,2,:])
                else:
                    mu_adjacent = mu_adjacent.set(csdl.slice[:,:,2], value=mu_grid[:,:,i,j-1])
                    mu_adjacent = mu_adjacent.set(csdl.slice[:,:,3], value=mu_grid[:,:,i,j+1])

                    deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,2,:], value=delta_coll_point[:,:,i,j,2,:])
                    deltas_sub_matrix = deltas_sub_matrix.set(csdl.slice[:,:,3,:], value=delta_coll_point[:,:,i,j,3,:])

            sub_delta_mu_vec = mu_adjacent - mu_panel_exp
            
            delta_system_matrix = delta_system_matrix.set(csdl.slice[:,:,start:stop,2*panel_ind:2*(panel_ind+1)], value=deltas_sub_matrix)
            delta_mu_vec = delta_mu_vec.set(csdl.slice[:,:,start:stop], value=sub_delta_mu_vec)
            
            start += deltas_per_panel_grid[i,j]
            panel_ind += 1

    print('computing transposes')
    a = time.time()
    delta_system_matrix_T = csdl.einsum(delta_system_matrix, action='ijkl->ijlk')
    b = time.time()
    print(f'transpose einsum time: {b-a} seconds')

    print('setting up linear system matrix')
    lin_sys_matrix = csdl.einsum(delta_system_matrix_T, delta_system_matrix, action='ijkl,ijlm->ijkm')
    c = time.time()
    print(f'matrix matrix product for linear system time: {c-b} seconds')

    print('RHS matrix vector product')
    RHS = csdl.einsum(delta_system_matrix_T, delta_mu_vec, action='ijkl,ijl->ijk')
    d = time.time()
    print(f'RHS matrix vector product computation time: {d-c} seconds')
    dmu_d = csdl.Variable(shape=(num_nodes, nt, num_panels*2), value=0.)
    print('solving linear system for least squares')
    for i in csdl.frange(num_nodes):
        for j in csdl.frange(nt):
            dmu_d = dmu_d.set(csdl.slice[i,j,:], value=csdl.solve_linear(lin_sys_matrix[i,j,:,:], RHS[i,j,:]))

    ql = -dmu_d[:,:,0::2].reshape((num_nodes, nt, nc_panels, ns_panels))
    qm = -dmu_d[:,:,1::2].reshape((num_nodes, nt, nc_panels, ns_panels))
    return ql, qm

def least_squares_velocity(mu_grid, delta_coll_point):
    '''
    We use the normal equations to solve for the derivative approximations, skipping the assembly of the original matrices
    A^{T}Ax   = A^{T}b becomes Cx = d; we generate C and d directly
    '''
    num_nodes, nt = mu_grid.shape[0], mu_grid.shape[1]
    grid_shape = mu_grid.shape[2:]
    nc_panels, ns_panels = grid_shape[0], grid_shape[1]
    num_panels = nc_panels*ns_panels
    C = csdl.Variable(shape=(num_nodes, nt, num_panels*2, num_panels*2), value=0.)
    b = csdl.Variable((num_nodes, nt, num_panels*2,), value=0.)

    # matrix assembly for C
    sum_dl_sq = csdl.sum(delta_coll_point[:,:,:,:,:,0]**2, axes=(4,))
    sum_dm_sq = csdl.sum(delta_coll_point[:,:,:,:,:,1]**2, axes=(4,))
    sum_dl_dm = csdl.sum(delta_coll_point[:,:,:,:,:,0]*delta_coll_point[:,:,:,:,:,1], axes=(4,))

    diag_list_dl = np.arange(start=0, stop=2*num_panels, step=2)
    diag_list_dm = diag_list_dl + 1
    # off_diag_indices = np.arange()

    C = C.set(csdl.slice[:,:,list(diag_list_dl), list(diag_list_dl)], value=sum_dl_sq.reshape((num_nodes, nt, num_panels)))
    C = C.set(csdl.slice[:,:,list(diag_list_dm), list(diag_list_dm)], value=sum_dm_sq.reshape((num_nodes, nt, num_panels)))
    C = C.set(csdl.slice[:,:,list(diag_list_dl), list(diag_list_dm)], value=sum_dl_dm.reshape((num_nodes, nt, num_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO
    C = C.set(csdl.slice[:,:,list(diag_list_dm), list(diag_list_dl)], value=sum_dl_dm.reshape((num_nodes, nt, num_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO

    # vector assembly for d
    mu_grid_exp_deltas = csdl.expand(mu_grid, mu_grid.shape + (4,), 'ijkl->ijkla')

    dmu = csdl.Variable(shape=(num_nodes, nt, nc_panels, ns_panels, 4), value=0.)
    # the last dimension of size 4 is minus l, plus l, minus m, plus m
    dmu = dmu.set(csdl.slice[:,:,1:,:,0], value = mu_grid[:,:,:-1,:] - mu_grid[:,:,1:,:])
    dmu = dmu.set(csdl.slice[:,:,:-1,:,1], value = mu_grid[:,:,1:,:] - mu_grid[:,:,:-1,:])
    dmu = dmu.set(csdl.slice[:,:,:,1:,2], value = mu_grid[:,:,:,:-1] - mu_grid[:,:,:,1:])
    dmu = dmu.set(csdl.slice[:,:,:,:-1,3], value = mu_grid[:,:,:,1:] - mu_grid[:,:,:,:-1])

    dl_dot_dmu = csdl.sum(delta_coll_point[:,:,:,:,:,0] * dmu, axes=(4,)).reshape((num_nodes, nt, num_panels))
    dm_dot_dmu = csdl.sum(delta_coll_point[:,:,:,:,:,1] * dmu, axes=(4,)).reshape((num_nodes, nt, num_panels))

    b = b.set(csdl.slice[:,:,0::2], value=dl_dot_dmu)
    b = b.set(csdl.slice[:,:,1::2], value=dm_dot_dmu)

    dmu_d = csdl.Variable(shape=(num_nodes, nt, num_panels*2), value=0.)
    for i in csdl.frange(num_nodes):
        for j in csdl.frange(nt):
            dmu_d = dmu_d.set(csdl.slice[i,j,:], value=csdl.solve_linear(C[i,j,:], b[i,j,:]))

    ql = -dmu_d[:,:,0::2].reshape((num_nodes, nt, nc_panels, ns_panels))
    qm = -dmu_d[:,:,1::2].reshape((num_nodes, nt, nc_panels, ns_panels))

    return ql, qm
