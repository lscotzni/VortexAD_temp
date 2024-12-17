import numpy as np 
import csdl_alpha as csdl

import time

from VortexAD.core.panel_method.steady.least_squares_velocity import least_squares_velocity   

def post_processor(mesh_dict, mu, sigma, num_nodes):
    surface_names = list(mesh_dict.keys())
    start, stop = 0, 0
    x_dir_global = np.array([1., 0., 0.])
    z_dir_global = np.array([0., 0., 1.])

    output_dict = {}
    for i in range(len(surface_names)):
        surface_name = surface_names[i]
        surf_dict = {}

        num_panels = mesh_dict[surface_name]['num_panels']
        nc, ns = mesh_dict[surface_name]['nc'], mesh_dict[surface_name]['ns']
        stop += num_panels

        mu_grid = mu[:,start:stop].reshape((num_nodes, nc-1, ns-1))

        # perturbation velocities
        qn = sigma[:,start:stop].reshape((num_nodes, nc-1, ns-1)) # num_nodes, nt, num_panels for surface

        # region least squares method for perturbation velocities (derivatives)
        delta_coll_point = mesh_dict[surface_name]['delta_coll_point']
        ql, qm = least_squares_velocity(mu_grid, delta_coll_point)
        # endregion

        panel_x_dir = mesh_dict[surface_name]['panel_x_dir']
        panel_y_dir = mesh_dict[surface_name]['panel_y_dir']
        panel_normal = mesh_dict[surface_name]['panel_normal']
        nodal_cp_velocity = mesh_dict[surface_name]['nodal_cp_velocity']
        coll_vel = mesh_dict[surface_name]['coll_point_velocity']
        if coll_vel:
            total_vel = nodal_cp_velocity+coll_vel
        else:
            total_vel = nodal_cp_velocity

        free_stream_l = csdl.einsum(total_vel, panel_x_dir, action='jklm,jklm->jkl')
        free_stream_m = csdl.einsum(total_vel, panel_y_dir, action='jklm,jklm->jkl')
        free_stream_n = csdl.einsum(total_vel, panel_normal, action='jklm,jklm->jkl')
        # print(mu_grid[0,0,:,:].value)
        # print(ql[0,0,:,:].value)
        # exit()
        Ql = free_stream_l + ql
        Qm = free_stream_m + qm
        Qn = free_stream_n + qn
        Q_inf_norm = csdl.norm(total_vel, axes=(3,))

        body_vel = csdl.Variable(shape=Ql.shape + (3,), value=0.)
        body_vel = body_vel.set(csdl.slice[:,:,:,0], value=Ql)
        body_vel = body_vel.set(csdl.slice[:,:,:,1], value=Qm)
        body_vel = body_vel.set(csdl.slice[:,:,:,2], value=Qn)
        body_vel_norm = csdl.norm(body_vel, axes=(3,))

        rot_mat = mesh_dict[surface_name]['rot_mat']
        body_vel_global = csdl.einsum(body_vel, rot_mat, action='jklm,jklmn->jkln')


        perturbed_vel_mag = (Ql**2 + Qm**2 + Qn**2)**0.5 
        Cp_static = 1 - (Ql**2 + Qm**2 + Qn**2)/Q_inf_norm**2
        Cp = Cp_static
        
        panel_area = mesh_dict[surface_name]['panel_area']

        rho = 1.225
        # rho = 1000.
        dP = -0.5*rho*Q_inf_norm**2*Cp
        dF_no_normal = dP*panel_area
        dF = csdl.expand(dF_no_normal, panel_normal.shape, 'jkl->jkla') * panel_normal

        Fz_panel = csdl.tensordot(dF, z_dir_global, axes=([3],[0]))
        Fx_panel = csdl.tensordot(dF, x_dir_global, axes=([3],[0]))

        nc_panels = int(num_panels/(ns-1))

        LE_velocity = (total_vel[:,int((nc_panels/2)-1),:,:] + total_vel[:,int(nc_panels/2),:,:])/2.
        aoa = csdl.arctan(LE_velocity[:,:,2]/LE_velocity[:,:,0])

        aoa_exp = csdl.expand(aoa, Fz_panel.shape, 'jk->jak')

        cosa, sina = csdl.cos(aoa_exp), csdl.sin(aoa_exp)

        panel_L = Fz_panel*cosa - Fx_panel*sina
        panel_Di = Fz_panel*sina + Fx_panel*cosa

        L = csdl.sum(panel_L, axes=(1,2))
        Di = csdl.sum(panel_Di, axes=(1,2))

        Q_inf = csdl.norm(csdl.average(LE_velocity, axes=(1,)), axes=(1,))

        planform_area = mesh_dict[surface_name]['planform_area']
        CL = L/(0.5*rho*planform_area*Q_inf**2)
        CDi = Di/(0.5*rho*planform_area*Q_inf**2)

        surf_dict['Cp'] = Cp
        surf_dict['CL'] = CL
        surf_dict['CDi'] = CDi
        surf_dict['Fx_panel'] = Fx_panel
        surf_dict['Fz_panel'] = Fz_panel
        surf_dict['panel_forces'] = dF
        
        surf_dict['body_vel'] = body_vel_norm

        surf_dict['panel_pressure'] = dP
        surf_dict['surface_vel'] = body_vel_global

        start += num_panels
        
        output_dict[surface_name] = surf_dict

        # print(CL.value)
        # print(CDi.value)

    return output_dict

