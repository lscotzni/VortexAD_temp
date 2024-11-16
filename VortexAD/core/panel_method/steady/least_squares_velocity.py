import numpy as np
import csdl_alpha as csdl 

def least_squares_velocity(mu_grid, delta_coll_point):
    '''
    We use the normal equations to solve for the derivative approximations, skipping the assembly of the original matrices
    A^{T}Ax   = A^{T}b becomes Cx = d; we generate C and d directly
    '''
    num_nodes= mu_grid.shape[0]
    grid_shape = mu_grid.shape[1:]
    nc_panels, ns_panels = grid_shape[0], grid_shape[1]
    num_panels = nc_panels*ns_panels
    C = csdl.Variable(shape=(num_nodes, num_panels*2, num_panels*2), value=0.)
    b = csdl.Variable((num_nodes, num_panels*2,), value=0.)

    # matrix assembly for C
    sum_dl_sq = csdl.sum(delta_coll_point[:,:,:,:,0]**2, axes=(3,))
    sum_dm_sq = csdl.sum(delta_coll_point[:,:,:,:,1]**2, axes=(3,))
    sum_dl_dm = csdl.sum(delta_coll_point[:,:,:,:,0]*delta_coll_point[:,:,:,:,1], axes=(3,))

    diag_list_dl = np.arange(start=0, stop=2*num_panels, step=2)
    diag_list_dm = diag_list_dl + 1
    # off_diag_indices = np.arange()

    C = C.set(csdl.slice[:,list(diag_list_dl), list(diag_list_dl)], value=sum_dl_sq.reshape((num_nodes, num_panels)))
    C = C.set(csdl.slice[:,list(diag_list_dm), list(diag_list_dm)], value=sum_dm_sq.reshape((num_nodes, num_panels)))
    C = C.set(csdl.slice[:,list(diag_list_dl), list(diag_list_dm)], value=sum_dl_dm.reshape((num_nodes, num_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO
    C = C.set(csdl.slice[:,list(diag_list_dm), list(diag_list_dl)], value=sum_dl_dm.reshape((num_nodes, num_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO

    # vector assembly for d
    mu_grid_exp_deltas = csdl.expand(mu_grid, mu_grid.shape + (4,), 'jkl->jkla')

    dmu = csdl.Variable(shape=(num_nodes, nc_panels, ns_panels, 4), value=0.)
    # the last dimension of size 4 is minus l, plus l, minus m, plus m
    dmu = dmu.set(csdl.slice[:,1:,:,0], value = mu_grid[:,:-1,:] - mu_grid[:,1:,:])
    dmu = dmu.set(csdl.slice[:,:-1,:,1], value = mu_grid[:,1:,:] - mu_grid[:,:-1,:])
    dmu = dmu.set(csdl.slice[:,:,1:,2], value = mu_grid[:,:,:-1] - mu_grid[:,:,1:])
    dmu = dmu.set(csdl.slice[:,:,:-1,3], value = mu_grid[:,:,1:] - mu_grid[:,:,:-1])

    dl_dot_dmu = csdl.sum(delta_coll_point[:,:,:,:,0] * dmu, axes=(3,)).reshape((num_nodes, num_panels))
    dm_dot_dmu = csdl.sum(delta_coll_point[:,:,:,:,1] * dmu, axes=(3,)).reshape((num_nodes, num_panels))

    b = b.set(csdl.slice[:,0::2], value=dl_dot_dmu)
    b = b.set(csdl.slice[:,1::2], value=dm_dot_dmu)

    dmu_d = csdl.Variable(shape=(num_nodes, num_panels*2), value=0.)
    for i in csdl.frange(num_nodes):
        dmu_d = dmu_d.set(csdl.slice[i,:], value=csdl.solve_linear(C[i,:], b[i,:]))

    ql = -dmu_d[:,0::2].reshape((num_nodes, nc_panels, ns_panels))
    qm = -dmu_d[:,1::2].reshape((num_nodes, nc_panels, ns_panels))

    return ql, qm