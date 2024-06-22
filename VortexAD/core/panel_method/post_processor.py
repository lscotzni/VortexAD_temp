import numpy as np 
import csdl_alpha as csdl

import time

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

def least_squares_velocity(mu_grid, delta_coll_point):
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
            
            mu_grid[:,:,i,j]
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
    

def post_processor(mesh_dict, mu, sigma, num_nodes, nt, dt):
    surface_names = list(mesh_dict.keys())
    start, stop = 0, 0
    x_dir_global = np.array([1., 0., 0.])
    z_dir_global = np.array([0., 0., 1.])

    output_dict = {}
    for i in range(len(surface_names)):
        surf_dict = {}
        surface_name = surface_names[i]
        num_panels = mesh_dict[surface_name]['num_panels']
        nc, ns = mesh_dict[surface_name]['nc'], mesh_dict[surface_name]['ns']
        stop += num_panels

        mu_grid = mu[:,:,start:stop].reshape((num_nodes, nt, nc-1, ns-1))

        # perturbation velocities
        qn = -sigma[:,:,start:stop].reshape((num_nodes, nt, nc-1, ns-1)) # num_nodes, nt, num_panels for surface

        pos_dl_norm, neg_dl_norm = mesh_dict[surface_name]['panel_dl_norm']
        pos_dm_norm, neg_dm_norm = mesh_dict[surface_name]['panel_dm_norm']

        if True:
            # region least squares method for derivatives
            delta_coll_point = mesh_dict[surface_name]['delta_coll_point']
            ql, qm = least_squares_velocity(mu_grid, delta_coll_point)
            # endregion
        if False:
            # region updated fd method
            # FUTURE FIX: MAKE A GRID AT THE CENTER OF PANEL EDGES THAT HOLDS A DL BETWEEN CORRESPONDING PANEL CENTERS
            # WE SAVE 2X INFORMATION THIS WAY
            dl = csdl.Variable(shape=pos_dl_norm.shape + (2,), value=0.) # last dimension is (negative dl, positive dl)
            dl = dl.set(csdl.slice[:,:,1:,:,0], value=neg_dl_norm[:,:,1:,:] + pos_dl_norm[:,:,:-1,:])
            dl = dl.set(csdl.slice[:,:,0,:,0], value=neg_dl_norm[:,:,0,:] + pos_dl_norm[:,:,-1])
            dl = dl.set(csdl.slice[:,:,:-1,:,1], value=pos_dl_norm[:,:,:-1,:] + neg_dl_norm[:,:,1:,:])
            dl = dl.set(csdl.slice[:,:,-1,:,1], value=pos_dl_norm[:,:,-1,:] + neg_dl_norm[:,:,0,:])

            dm = csdl.Variable(shape=pos_dm_norm.shape + (2,), value=0.)
            dm = dm.set(csdl.slice[:,:,:,1:,0], value=pos_dm_norm[:,:,:,:-1] + neg_dm_norm[:,:,:,1:])
            dm = dm.set(csdl.slice[:,:,:,:-1,1], value=neg_dm_norm[:,:,:,1:] + pos_dm_norm[:,:,:,:-1])

            ql, qm = perturbation_velocity_FD(mu_grid, dl, dm)
            # endregion

        if False:
            # region original fd method
            panel_center = mesh_dict[surface_name]['panel_center'] # nn, nt, nc-1, ns-1, 3

            panel_center_dl_magnitude = csdl.norm(panel_center[:,:,1:,:,:] - panel_center[:,:,:-1,:,:], axes=(4,))
            panel_center_dm_magnitude = csdl.norm(panel_center[:,:,:,1:,:] - panel_center[:,:,:,:-1,:], axes=(4,))

            panel_center_dl = csdl.Variable(shape=(num_nodes, nt, nc-1, ns-1, 2), value=0.)
            panel_center_dm = csdl.Variable(shape=panel_center_dl.shape, value=0.)

            panel_center_dl = panel_center_dl.set(csdl.slice[:,:,:-1,:,0], value=panel_center_dl_magnitude)
            panel_center_dl = panel_center_dl.set(csdl.slice[:,:,1:,:,1], value=panel_center_dl_magnitude)

            panel_center_dm = panel_center_dm.set(csdl.slice[:,:,:,:-1,0], value=panel_center_dm_magnitude)
            panel_center_dm = panel_center_dm.set(csdl.slice[:,:,:,1:,1], value=panel_center_dm_magnitude)

            ql = csdl.Variable(shape=mu_grid.shape, value=0.)
            qm = csdl.Variable(shape=mu_grid.shape, value=0.)

            ql = ql.set(csdl.slice[:,:,1:-1,:], value=-(mu_grid[:,:,2:,:] - mu_grid[:,:,:-2,:])/2./((panel_center_dl[:,:,1:-1,:,0]+panel_center_dl[:,:,1:-1,:,1])/2))
            ql = ql.set(csdl.slice[:,:,0,:], value=-(-3*mu_grid[:,:,0,:]+4*mu_grid[:,:,1,:]-mu_grid[:,:,2,:])/2./((panel_center_dl[:,:,0,:,0]+panel_center_dl[:,:,1,:,0])/2))
            ql = ql.set(csdl.slice[:,:,-1,:], value=-(3*mu_grid[:,:,-1,:]-4*mu_grid[:,:,-2,:]+mu_grid[:,:,-3,:])/2./((panel_center_dl[:,:,-1,:,1]+panel_center_dl[:,:,-2,:,1])/2))

            qm = qm.set(csdl.slice[:,:,:,1:-1], value=-(mu_grid[:,:,:,2:] - mu_grid[:,:,:,:-2])/2./((panel_center_dm[:,:,:,1:-1,0]+panel_center_dm[:,:,:,1:-1,1])/2))
            qm = qm.set(csdl.slice[:,:,:,0], value=-(-3*mu_grid[:,:,:,0]+4*mu_grid[:,:,:,1]-mu_grid[:,:,:,2])/2./((panel_center_dm[:,:,:,0,0]+panel_center_dm[:,:,:,1,0])/2))
            qm = qm.set(csdl.slice[:,:,:,-1], value=-(3*mu_grid[:,:,:,-1]-4*mu_grid[:,:,:,-2]+mu_grid[:,:,:,-3])/2./((panel_center_dm[:,:,:,-1,1]+panel_center_dm[:,:,:,-2,1])/2))
            # endregion

        panel_x_dir = mesh_dict[surface_name]['panel_x_dir']
        panel_y_dir = mesh_dict[surface_name]['panel_y_dir']
        panel_normal = mesh_dict[surface_name]['panel_normal']
        coll_vel = mesh_dict[surface_name]['coll_point_velocity']

        free_stream_l = csdl.einsum(coll_vel, panel_x_dir, action='ijklm,ijklm->ijkl')
        free_stream_m = csdl.einsum(coll_vel, panel_y_dir, action='ijklm,ijklm->ijkl')
        free_stream_n = csdl.einsum(coll_vel, panel_normal, action='ijklm,ijklm->ijkl')

        Ql = free_stream_l + ql
        Qm = free_stream_m + qm
        Qn = free_stream_n + qn
        Q_inf_norm = csdl.norm(coll_vel, axes=(4,))

        dmu_dt = csdl.Variable(shape=Q_inf_norm.shape, value=0)
        if nt > 2:
            dmu_dt = dmu_dt.set(csdl.slice[:,1:,:,:], value=(mu_grid[:,1:,:,:] - mu_grid[:,:-1,:,:])/dt)

        perturbed_vel_mag = (Ql**2 + Qm**2 + Qn**2)**0.5 
        Cp = 1 - (Ql**2 + Qm**2 + Qn**2)/Q_inf_norm**2 - dmu_dt*2./Q_inf_norm**2
        # Cp = 1 - (Ql**2 + Qn**2)/Q_inf_norm**2 - dmu_dt*2./Q_inf_norm**2

        panel_area = mesh_dict[surface_name]['panel_area']

        rho = 1.22506547
        dF_no_normal = -0.5*rho*Q_inf_norm**2*panel_area*Cp
        dF = csdl.expand(dF_no_normal, panel_normal.shape, 'ijkl->ijkla') * panel_normal

        Fz_panel = csdl.tensordot(dF, z_dir_global, axes=([4],[0]))
        Fx_panel = csdl.tensordot(dF, x_dir_global, axes=([4],[0]))

        surface_area = csdl.sum(panel_area, axes=(2,3))

        nc_panels = int(num_panels/(ns-1))

        LE_velocity = (coll_vel[:,:,int((nc_panels/2)-1),:,:] + coll_vel[:,:,int(nc_panels/2),:,:])/2.
        aoa = csdl.arctan(LE_velocity[:,:,:,2]/LE_velocity[:,:,:,0])

        aoa_exp = csdl.expand(aoa, Fz_panel.shape, 'ijk->ijak')

        cosa, sina = csdl.cos(aoa_exp), csdl.sin(aoa_exp)

        panel_L = Fz_panel*cosa - Fx_panel*sina
        panel_Di = Fz_panel*sina + Fx_panel*cosa

        # Fz = csdl.sum(Fz_panel, axes=(2,3))
        # Fx = csdl.sum(Fx_panel, axes=(2,3))

        L = csdl.sum(panel_L, axes=(2,3))
        Di = csdl.sum(panel_Di, axes=(2,3))

        Q_inf = csdl.norm(csdl.average(LE_velocity, axes=(2,)), axes=(2,))

        CL = L/(0.5*rho*surface_area*Q_inf**2)
        CDi = Di/(0.5*rho*surface_area*Q_inf**2)

        surf_dict['Cp'] = Cp
        surf_dict['CL'] = CL
        output_dict[surface_name] = surf_dict

        print(CL.value)
        print(CDi.value)


    return output_dict