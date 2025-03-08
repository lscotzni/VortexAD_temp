import numpy as np 
import csdl_alpha as csdl

import time

from VortexAD.core.panel_method.steady.least_squares_velocity import least_squares_velocity, unstructured_least_squares_velocity
from VortexAD.core.panel_method.steady.least_squares_velocity import least_squares_velocity_old

def post_processor(mesh_dict, mu, sigma, num_nodes, rho=1.225):
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
        # ql, qm = least_squares_velocity_old(mu_grid, delta_coll_point)
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
        surf_dict['L'] = L
        
        surf_dict['body_vel'] = body_vel_norm

        surf_dict['panel_pressure'] = dP
        surf_dict['surface_vel'] = body_vel_global

        start += num_panels
        
        output_dict[surface_name] = surf_dict

        # print(CL.value)
        # print(CDi.value)

    return output_dict


def unstructured_post_processor(mesh_dict, mu, sigma, num_nodes, rho=1.225):
    x_dir_global = np.array([1., 0., 0.])
    z_dir_global = np.array([0., 0., 1.])
    output_dict = {}

    qn = sigma
    delta_coll_point = mesh_dict['delta_coll_point']
    cell_adjacency = mesh_dict['cell_adjacency']

    ql, qm = unstructured_least_squares_velocity(mu, delta_coll_point, cell_adjacency)

    panel_x_dir = mesh_dict['panel_x_dir']
    panel_y_dir = mesh_dict['panel_y_dir']
    panel_normal = mesh_dict['panel_normal']
    coll_vel = mesh_dict['coll_point_velocity']

    free_stream_l = csdl.einsum(coll_vel, panel_x_dir, action='jkl,jkl->jk')
    free_stream_m = csdl.einsum(coll_vel, panel_y_dir, action='jkl,jkl->jk')
    free_stream_n = csdl.einsum(coll_vel, panel_normal, action='jkl,jkl->jk')

    Ql = free_stream_l + ql
    Qm = free_stream_m + qm
    Qn = free_stream_n + qn
    Q_inf_norm = csdl.norm(coll_vel, axes=(2,))
    
    perturbed_vel_mag = (Ql**2 + Qm**2 + Qn**2)**0.5
    Cp_static = 1 - perturbed_vel_mag**2/Q_inf_norm**2
    # Cp_dynamic = -dmu_dt*2./Q_inf_norm**2
    Cp = Cp_static
    # Cp = csdl.maximum(Cp, -10*np.ones(shape=Cp.shape))

    panel_area = mesh_dict['panel_area']
    dF_no_normal = -0.5*rho*Q_inf_norm**2*panel_area*Cp
    dF = csdl.expand(dF_no_normal, panel_normal.shape, 'jk->jka')*panel_normal
    Fz_panel = csdl.tensordot(dF, z_dir_global, axes=([2],[0]))
    Fx_panel = csdl.tensordot(dF, x_dir_global, axes=([2],[0]))

    aoa = csdl.arctan(coll_vel[:,:,2]/coll_vel[:,:,0])
    cosa, sina = csdl.cos(aoa), csdl.sin(aoa)

    panel_L = Fz_panel*cosa - Fx_panel*sina
    panel_Di = Fz_panel*sina + Fx_panel*cosa

    L = csdl.sum(panel_L, axes=(1,))
    Di = csdl.sum(panel_Di, axes=(1,))

    Q_inf = csdl.average(Q_inf_norm, axes=(1,))

    ref_area = 10.
    ref_area = 507.610
    CL = L/(0.5*rho*ref_area*Q_inf**2)
    CDi = Di/(0.5*rho*ref_area*Q_inf**2)

    output_dict['CL'] = CL
    output_dict['CDi'] = CDi
    output_dict['Cp'] = Cp
    output_dict['panel_forces'] = dF
    output_dict['Qn'] = Qn
    # output_dict['Ql'] = Ql
    output_dict['L'] = L

    return output_dict