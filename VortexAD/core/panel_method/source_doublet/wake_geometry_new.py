import numpy as np 
import csdl_alpha as csdl

from VortexAD.utils.atan2_switch import atan2_switch

def wake_geometry(surf_wake_mesh_dict, time_ind):

    mesh = surf_wake_mesh_dict['mesh']
    mesh_shape = mesh.shape
    nt, ns = mesh_shape[-3], mesh_shape[-2]
    num_panels = surf_wake_mesh_dict['num_panels']
    # surf_wake_mesh_dict['num_points'] = nt*ns

    # surf_wake_mesh_dict['num_panels'] = (nt-1)*(ns-1)
    # surf_wake_mesh_dict['nt'] = nt
    # surf_wake_mesh_dict['ns'] = ns

    # p1 = mesh[:,time_ind,:-1,:-1,:]
    # p2 = mesh[:,time_ind,:-1,1:,:]
    # p3 = mesh[:,time_ind,1:,1:,:]
    # p4 = mesh[:,time_ind,1:,:-1,:]

    R1 = mesh[:,time_ind,:-1,:-1,:]
    R2 = mesh[:,time_ind,1:,:-1,:]
    R3 = mesh[:,time_ind,1:,1:,:]
    R4 = mesh[:,time_ind,:-1,1:,:]

    panel_center = surf_wake_mesh_dict['panel_center']
    Rc = (R1+R2+R3+R4)/4.
    panel_center = panel_center.set(csdl.slice[:,time_ind,:,:,:], value=Rc)
    surf_wake_mesh_dict['panel_center'] = panel_center

    panel_corners = surf_wake_mesh_dict['panel_corners']
    panel_corners = panel_corners.set(csdl.slice[:,time_ind,:,:,0,:], value=R1)
    panel_corners = panel_corners.set(csdl.slice[:,time_ind,:,:,1,:], value=R2)
    panel_corners = panel_corners.set(csdl.slice[:,time_ind,:,:,2,:], value=R3)
    panel_corners = panel_corners.set(csdl.slice[:,time_ind,:,:,3,:], value=R4)
    surf_wake_mesh_dict['panel_corners'] = panel_corners

    D1 = R3-R1
    D2 = R4-R2

    D1D2_cross = csdl.cross(D1, D2, axis=3)
    D1D2_cross_norm = csdl.norm(D1D2_cross+1.e-12, axes=(3,))
    panel_area = surf_wake_mesh_dict['panel_area']
    panel_area = panel_area.set(csdl.slice[:,time_ind,:,:], value=D1D2_cross_norm/2.)
    surf_wake_mesh_dict['panel_area'] = panel_area

    panel_x_dir = surf_wake_mesh_dict['panel_x_dir']
    panel_y_dir = surf_wake_mesh_dict['panel_y_dir']
    panel_normal = surf_wake_mesh_dict['panel_normal']

    normal_vec = D1D2_cross / csdl.expand(D1D2_cross_norm, D1D2_cross.shape, 'ijk->ijka')

    m_dir = (R3+R4)/2. - Rc
    m_norm = csdl.norm(m_dir, axes=(3,))
    m_vec = m_dir / csdl.expand(m_norm, m_dir.shape, 'ijk->ijka')
    l_vec = csdl.cross(m_vec, normal_vec, axis=3)

    panel_x_dir = panel_x_dir.set(csdl.slice[:,time_ind,:,:,:], value=l_vec)
    panel_y_dir = panel_y_dir.set(csdl.slice[:,time_ind,:,:,:], value=m_vec)
    panel_normal = panel_normal.set(csdl.slice[:,time_ind,:,:,:], value=normal_vec)

    surf_wake_mesh_dict['panel_x_dir'] = panel_x_dir
    surf_wake_mesh_dict['panel_y_dir'] = panel_y_dir
    surf_wake_mesh_dict['panel_normal'] = panel_normal

    S = surf_wake_mesh_dict['S']
    SL = surf_wake_mesh_dict['SL']
    SM = surf_wake_mesh_dict['SM']

    s = csdl.Variable(shape=(panel_corners.shape[0],) + panel_corners.shape[2:], value=0.)
    s = s.set(csdl.slice[:,:,:,:-1,:], value=panel_corners[:,time_ind,:,:,1:,:] - panel_corners[:,time_ind,:,:,:-1,:])
    s = s.set(csdl.slice[:,:,:,-1,:], value=panel_corners[:,time_ind,:,:,0,:] - panel_corners[:,time_ind,:,:,-1,:])

    l_exp = csdl.expand(l_vec, s.shape, 'ijkl->ijkal')
    m_exp = csdl.expand(m_vec, s.shape, 'ijkl->ijkal')
    
    S_val = csdl.norm(s+1.e-12, axes=(4,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
    SL_val = csdl.sum(s*l_exp+1.e-12, axes=(4,))
    SM_val = csdl.sum(s*m_exp+1.e-12, axes=(4,))

    S = S.set(csdl.slice[:,time_ind,:,:,:], value=S_val)
    SL = SL.set(csdl.slice[:,time_ind,:,:,:], value=SL_val)
    SM = SM.set(csdl.slice[:,time_ind,:,:,:], value=SM_val)

    surf_wake_mesh_dict['S'] = S
    surf_wake_mesh_dict['SL'] = SL
    surf_wake_mesh_dict['SM'] = SM

    print(S.shape)
    # exit()


    # dpij_global = csdl.Variable(value=np.zeros(((panel_center.shape[0],) + (panel_center.shape[2:4]) + (4,3))))
    # dpij_global = dpij_global.set(csdl.slice[:,:,:,0,:], value=p2[:,:,:,:]-p1[:,:,:,:])
    # dpij_global = dpij_global.set(csdl.slice[:,:,:,1,:], value=p3[:,:,:,:]-p2[:,:,:,:])
    # dpij_global = dpij_global.set(csdl.slice[:,:,:,2,:], value=p4[:,:,:,:]-p3[:,:,:,:])
    # dpij_global = dpij_global.set(csdl.slice[:,:,:,3,:], value=p1[:,:,:,:]-p4[:,:,:,:])

    # local_coord_vec = csdl.Variable(shape=((panel_center.shape[0],) + (panel_center.shape[2:4]) + (3,3)), value=0.) # last two indices are for 3 vectors, 3 dimensions
    # local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,0,:], value=panel_x_dir_val)
    # local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,1,:], value=panel_y_dir_val)
    # local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,2,:], value=panel_normal_val)

    # nodal_vel = surf_wake_mesh_dict[key]['nodal_velocity']
    # surf_wake_mesh_dict[key]['coll_point_velocity'] = (nodal_vel[:,:,:-1,:-1,:]+nodal_vel[:,:,:-1,1:,:]+nodal_vel[:,:,1:,1:,:]+nodal_vel[:,:,1:,:-1,:]) / 4.

    return surf_wake_mesh_dict