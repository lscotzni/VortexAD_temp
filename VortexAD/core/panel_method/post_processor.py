import numpy as np 
import csdl_alpha as csdl

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

        panel_x_dir = mesh_dict[surface_name]['panel_x_dir']
        panel_y_dir = mesh_dict[surface_name]['panel_y_dir']
        panel_normal = mesh_dict[surface_name]['panel_normal']
        coll_vel = mesh_dict[surface_name]['coll_point_velocity']

        free_stream_l = csdl.einsum(-coll_vel, panel_x_dir, action='ijklm,ijklm->ijkl')
        free_stream_m = csdl.einsum(-coll_vel, panel_y_dir, action='ijklm,ijklm->ijkl')
        free_stream_n = csdl.einsum(-coll_vel, panel_normal, action='ijklm,ijklm->ijkl')

        Ql = free_stream_l + ql
        Qm = free_stream_m + qm
        Qn = free_stream_n + qn
        Q_inf_norm = csdl.norm(coll_vel, axes=(4,))

        dmu_dt = csdl.Variable(shape=Q_inf_norm.shape, value=0)
        if nt > 2:
            dmu_dt = dmu_dt.set(csdl.slice[:,1:,:,:], value=(mu_grid[:,1:,:,:] - mu_grid[:,:-1,:,:])/dt)

        Cp = 1 - (Ql**2 + Qm**2 + Qn**2)/Q_inf_norm**2 - dmu_dt*2./Q_inf_norm**2
        # Cp = 1 - (Ql**2 + Qn**2)/Q_inf_norm**2 - dmu_dt*2./Q_inf_norm**2

        panel_area = mesh_dict[surface_name]['panel_area']

        rho = 1.225
        dF_no_normal = -0.5*rho*Q_inf_norm**2*panel_area*Cp
        dF = csdl.expand(dF_no_normal, panel_normal.shape, 'ijkl->ijkla') * panel_normal

        Fz = csdl.tensordot(dF, z_dir_global, axes=([4],[0]))
        Fx = csdl.tensordot(dF, x_dir_global, axes=([4],[0]))

        surface_area = csdl.sum(panel_area, axes=(2,3))

        nc_panels = int(num_panels/(ns-1))

        LE_velocity = (coll_vel[:,:,int((nc_panels/2)-1),:,:] + coll_vel[:,:,int(nc_panels/2),:,:])/2.
        aoa = csdl.arctan(LE_velocity[:,:,:,2]/LE_velocity[:,:,:,0])

        aoa_exp = csdl.expand(aoa, Fz.shape, 'ijk->ijak')

        cosa, sina = csdl.cos(aoa_exp), csdl.sin(aoa_exp)

        panel_L = Fz*cosa - Fx*sina
        panel_Di = Fz*sina + Fx*cosa

        L = csdl.sum(panel_L, axes=(2,3))
        Di = csdl.sum(panel_Di, axes=(2,3))

        Q_inf = csdl.norm(csdl.average(LE_velocity, axes=(2,)), axes=(2,))

        CL = L/(0.5*rho*surface_area*Q_inf**2)
        CDi = Di/(0.5*rho*surface_area*Q_inf**2)

        surf_dict['Cp'] = Cp
        output_dict[surface_name] = surf_dict

    


    return output_dict