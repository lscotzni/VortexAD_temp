import numpy as np 
import csdl_alpha as csdl

import time

from VortexAD.core.panel_method.perturbation_velocity_comp import perturbation_velocity_FD, perturbation_velocity_FD_K_P
from VortexAD.core.panel_method.perturbation_velocity_comp import least_squares_velocity, unstructured_least_squares_velocity    

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

        panel_areas = mesh_dict[surface_name]['panel_area']

        mu_grid = mu[:,:,start:stop].reshape((num_nodes, nt, nc-1, ns-1))

        # perturbation velocities
        qn = -sigma[:,:,start:stop].reshape((num_nodes, nt, nc-1, ns-1)) # num_nodes, nt, num_panels for surface

        pos_dl_norm, neg_dl_norm = mesh_dict[surface_name]['panel_dl_norm']
        pos_dm_norm, neg_dm_norm = mesh_dict[surface_name]['panel_dm_norm']

        if True:
            # region least squares method for perturbation velocities (derivatives)
            delta_coll_point = mesh_dict[surface_name]['delta_coll_point']
            # ql, qm = least_squares_velocity_old(mu_grid, delta_coll_point)
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

            ql, qm = perturbation_velocity_FD_K_P(mu_grid, panel_center_dl, panel_center_dm)
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
            dmu_dt = dmu_dt.set(csdl.slice[:,1:,:,:], value=(mu_grid[:,:-1,:,:] - mu_grid[:,1:,:,:])/dt)

        perturbed_vel_mag = (Ql**2 + Qm**2 + Qn**2)**0.5 
        Cp_static = 1 - (Ql**2 + Qm**2 + Qn**2)/Q_inf_norm**2
        Cp_dynamic = -dmu_dt*2./Q_inf_norm**2
        Cp = Cp_static + Cp_dynamic
        # Cp = 1 - (Ql**2 + Qm**2 + Qn**2)/Q_inf_norm**2 - dmu_dt*2./Q_inf_norm**2
        # Cp = 1 - (Ql**2 + Qn**2)/Q_inf_norm**2 - dmu_dt*2./Q_inf_norm**2

        panel_area = mesh_dict[surface_name]['panel_area']

        rho = 1.225
        dF_no_normal = -0.5*rho*Q_inf_norm**2*panel_area*Cp
        dF = csdl.expand(dF_no_normal, panel_normal.shape, 'ijkl->ijkla') * panel_normal

        Fz_panel = csdl.tensordot(dF, z_dir_global, axes=([4],[0]))
        Fx_panel = csdl.tensordot(dF, x_dir_global, axes=([4],[0]))

        nc_panels = int(num_panels/(ns-1))

        LE_velocity = (coll_vel[:,:,int((nc_panels/2)-1),:,:] + coll_vel[:,:,int(nc_panels/2),:,:])/2.
        aoa = csdl.arctan(LE_velocity[:,:,:,2]/LE_velocity[:,:,:,0])

        aoa_exp = csdl.expand(aoa, Fz_panel.shape, 'ijk->ijak')

        cosa, sina = csdl.cos(aoa_exp), csdl.sin(aoa_exp)

        panel_L = Fz_panel*cosa - Fx_panel*sina
        panel_Di = Fz_panel*sina + Fx_panel*cosa

        L = csdl.sum(panel_L, axes=(2,3))
        Di = csdl.sum(panel_Di, axes=(2,3))

        Q_inf = csdl.norm(csdl.average(LE_velocity, axes=(2,)), axes=(2,))

        planform_area = mesh_dict[surface_name]['planform_area']
        CL = L/(0.5*rho*planform_area*Q_inf**2)
        CDi = Di/(0.5*rho*planform_area*Q_inf**2)

        surf_dict['Cp'] = Cp
        surf_dict['CL'] = CL
        surf_dict['CDi'] = CDi
        surf_dict['Fx_panel'] = Fx_panel
        surf_dict['Fz_panel'] = Fz_panel
        output_dict[surface_name] = surf_dict

        # print(CL.value)
        # print(CDi.value)


    return output_dict

def unstructured_post_processor(mesh_dict, mu, sigma, num_nodes, nt, dt):
    x_dir_global = np.array([1., 0., 0.])
    z_dir_global = np.array([0., 0., 1.])

    qn = -sigma
    delta_coll_point = mesh_dict['delta_coll_point']
    cell_adjacency = mesh_dict['cell_adjacency']

    ql, qm = unstructured_least_squares_velocity(mu, delta_coll_point, cell_adjacency)

    panel_x_dir = mesh_dict['panel_x_dir']
    panel_y_dir = mesh_dict['panel_y_dir']
    panel_normal = mesh_dict['panel_normal']
    coll_vel = mesh_dict['coll_point_velocity']

    free_stream_l = csdl.einsum(coll_vel, panel_x_dir, action='ijkl,ijkl->ijk')
    free_stream_m = csdl.einsum(coll_vel, panel_y_dir, action='ijkl,ijkl->ijk')
    free_stream_n = csdl.einsum(coll_vel, panel_normal, action='ijkl,ijkl->ijk')

    Ql = free_stream_l + ql
    Qm = free_stream_m + qm
    Qn = free_stream_n + qn
    Q_inf_norm = csdl.norm(coll_vel, axes=(3,))

    dmu_dt = csdl.Variable(shape=Q_inf_norm.shape, value=0.)
    if nt > 2:
        dmu_dt = dmu_dt.set(csdl.slice[:,1:,:], value=(mu[:,:-1,:] - mu[:,1:,:])/dt)
    
    perturbed_vel_mag = (Ql**2 + Qm**2 + Qn**2)**0.5
    Cp_static = 1 - perturbed_vel_mag**2/Q_inf_norm**2
    Cp_dynamic = -dmu_dt*2./Q_inf_norm**2
    Cp = Cp_static + Cp_dynamic

    panel_area = mesh_dict['panel_area']
    rho = 1.225
    dF_no_normal = -0.5*rho*Q_inf_norm**2*panel_area*Cp
    dF = csdl.expand(dF_no_normal, panel_normal.shape, 'ijk->ijka')*panel_normal
    Fz_panel = csdl.tensordot(dF, z_dir_global, axes=([3],[0]))
    Fx_panel = csdl.tensordot(dF, x_dir_global, axes=([3],[0]))


    aoa = csdl.arctan(coll_vel[:,:,:,2]/coll_vel[:,:,:,0])
    cosa, sina = csdl.cos(aoa), csdl.sin(aoa)

    panel_L = Fz_panel*cosa - Fx_panel*sina
    panel_Di = Fz_panel*sina + Fx_panel*cosa

    L = csdl.sum(panel_L, axes=(2,))
    Di = csdl.sum(panel_Di, axes=(2,))

    Q_inf = csdl.average(Q_inf_norm, axes=(2,))

    ref_area = 10.
    CL = L/(0.5*rho*ref_area*Q_inf**2)
    CDi = Di/(0.5*rho*ref_area*Q_inf**2)

    return