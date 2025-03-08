import numpy as np
import csdl_alpha as csdl

def post_processor():
    return

def unstructured_post_processor(mesh_dict, mu, sigma, rho=1.225):
    x_dir_global = np.array([1., 0., 0.])
    z_dir_global = np.array([0., 0., 1.])
    output_dict = {}
    
    qn = sigma
    ql = mu[:,:,1] # third column holds each component in the distribution
    qm = mu[:,:,2] # third column holds each component in the distribution
    # third column: constant, mu_x, mu_y, mu_xx, mu_yy, mu_xy, ...

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

    return