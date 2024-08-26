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

    p1 = mesh[:,time_ind,:-1,:-1,:]
    p2 = mesh[:,time_ind,1:,:-1,:]
    p3 = mesh[:,time_ind,1:,1:,:]
    p4 = mesh[:,time_ind,:-1,1:,:]

    panel_center = surf_wake_mesh_dict['panel_center']
    panel_center = panel_center.set(csdl.slice[:,time_ind,:,:,:], value=(p1+p2+p3+p4)/4.)
    surf_wake_mesh_dict['panel_center'] = panel_center

    panel_corners = surf_wake_mesh_dict['panel_corners']
    panel_corners = panel_corners.set(csdl.slice[:,time_ind,:,:,0,:], value=p1)
    panel_corners = panel_corners.set(csdl.slice[:,time_ind,:,:,1,:], value=p2)
    panel_corners = panel_corners.set(csdl.slice[:,time_ind,:,:,2,:], value=p3)
    panel_corners = panel_corners.set(csdl.slice[:,time_ind,:,:,3,:], value=p4)
    surf_wake_mesh_dict['panel_corners'] = panel_corners

    panel_area = surf_wake_mesh_dict['panel_area']
    panel_diag_1 = p3-p1
    # panel_diag_2 = p2-p4
    panel_diag_2 = p4-p2
    panel_normal_vec = csdl.cross(panel_diag_1, panel_diag_2, axis=3)
    panel_area = panel_area.set(csdl.slice[:,time_ind,:,:], value=csdl.norm(panel_normal_vec+1.e-12, axes=(3,)) / 2.)
    surf_wake_mesh_dict['panel_area'] = panel_area

    panel_x_dir = surf_wake_mesh_dict['panel_x_dir']
    panel_y_dir = surf_wake_mesh_dict['panel_y_dir']
    panel_normal = surf_wake_mesh_dict['panel_normal']

    # panel_x_vec = (p3+p4)/2. - (p1+p2)/2.
    # panel_y_vec = (p2+p3)/2. - (p1+p4)/2.

    panel_x_vec = (p3+p2)/2. - (p1+p4)/2.
    panel_y_vec = (p4+p3)/2. - (p1+p2)/2.

    panel_x_dir_val = panel_x_vec / csdl.expand(csdl.norm(panel_x_vec+1.e-12, axes=(3,)), panel_x_vec.shape, 'ijk->ijka')
    panel_y_dir_val = panel_y_vec / csdl.expand(csdl.norm(panel_y_vec+1.e-12, axes=(3,)), panel_y_vec.shape, 'ijk->ijka')
    panel_normal_val = panel_normal_vec / csdl.expand(csdl.norm(panel_normal_vec+1.e-12, axes=(3,)), panel_normal_vec.shape, 'ijk->ijka')

    panel_x_dir = panel_x_dir.set(csdl.slice[:,time_ind,:,:,:], value=panel_x_dir_val)
    panel_y_dir = panel_y_dir.set(csdl.slice[:,time_ind,:,:,:], value=panel_y_dir_val)
    panel_normal = panel_normal.set(csdl.slice[:,time_ind,:,:,:], value=panel_normal_val)

    surf_wake_mesh_dict['panel_x_dir'] = panel_x_dir
    surf_wake_mesh_dict['panel_y_dir'] = panel_y_dir
    surf_wake_mesh_dict['panel_normal'] = panel_normal

    # global unit vectors
    # +x points from tail to nose of aircraft
    # +y points to the left of the aircraft (facing from the front)
    # +z points down
    x_dir = np.array([1., 0., 0.,])
    y_dir = np.array([0., 1., 0.,])
    z_dir = np.array([0., 0., 1.,])

    Px_normal = csdl.tensordot(panel_normal_val, x_dir, axes=([3],[0]))
    Py_normal = csdl.tensordot(panel_normal_val, y_dir, axes=([3],[0]))
    Pz_normal = csdl.tensordot(panel_normal_val, z_dir, axes=([3],[0]))

    # theta_x = atan2_switch(Py_normal, Pz_normal, scale=100.) # roll angle
    # theta_y = atan2_switch(-Px_normal, Pz_normal, scale=100.) # pitch angle

    Px_x_dir = csdl.tensordot(panel_x_dir_val, x_dir, axes=([3],[0]))
    Py_x_dir = csdl.tensordot(panel_x_dir_val, y_dir, axes=([3],[0]))

    # theta_z = atan2_switch(Py_x_dir, Px_x_dir, scale=100.)
    # nn, nc, ns, 4, 3
    # dpij_global = csdl.Variable(value=np.zeros((panel_center.shape[:-1] + (4,3))))
    dpij_global = csdl.Variable(value=np.zeros(((panel_center.shape[0],) + (panel_center.shape[2:4]) + (4,3))))
    dpij_global = dpij_global.set(csdl.slice[:,:,:,0,:], value=p2[:,:,:,:]-p1[:,:,:,:])
    dpij_global = dpij_global.set(csdl.slice[:,:,:,1,:], value=p3[:,:,:,:]-p2[:,:,:,:])
    dpij_global = dpij_global.set(csdl.slice[:,:,:,2,:], value=p4[:,:,:,:]-p3[:,:,:,:])
    dpij_global = dpij_global.set(csdl.slice[:,:,:,3,:], value=p1[:,:,:,:]-p4[:,:,:,:])

    local_coord_vec = csdl.Variable(shape=((panel_center.shape[0],) + (panel_center.shape[2:4]) + (3,3)), value=0.) # last two indices are for 3 vectors, 3 dimensions
    local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,0,:], value=panel_x_dir_val)
    local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,1,:], value=panel_y_dir_val)
    local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,2,:], value=panel_normal_val)

    dpij_local = csdl.einsum(dpij_global, local_coord_vec, action='jklma,jklba->jklmb')  # THIS IS CORRECT
    
    dpij = surf_wake_mesh_dict['dpij']
    # dpij = dpij.set(csdl.slice[:,time_ind,:,:,0,:], value=p2[:,:,:,:2]-p1[:,:,:,:2])
    # dpij = dpij.set(csdl.slice[:,time_ind,:,:,1,:], value=p3[:,:,:,:2]-p2[:,:,:,:2])
    # dpij = dpij.set(csdl.slice[:,time_ind,:,:,2,:], value=p4[:,:,:,:2]-p3[:,:,:,:2])
    # dpij = dpij.set(csdl.slice[:,time_ind,:,:,3,:], value=p1[:,:,:,:2]-p4[:,:,:,:2])
    dpij = dpij.set(csdl.slice[:,time_ind,:,:,0,:], value=dpij_local[:,:,:,0,:2])
    dpij = dpij.set(csdl.slice[:,time_ind,:,:,1,:], value=dpij_local[:,:,:,1,:2])
    dpij = dpij.set(csdl.slice[:,time_ind,:,:,2,:], value=dpij_local[:,:,:,2,:2])
    dpij = dpij.set(csdl.slice[:,time_ind,:,:,3,:], value=dpij_local[:,:,:,3,:2])
    surf_wake_mesh_dict['dpij'] = dpij

    dij = surf_wake_mesh_dict['dij']
    dij = dij.set(csdl.slice[:,time_ind,:,:,0], value=csdl.norm(dpij[:,time_ind,:,:,0,:]+1.e-12, axes=(3,)))
    dij = dij.set(csdl.slice[:,time_ind,:,:,1], value=csdl.norm(dpij[:,time_ind,:,:,1,:]+1.e-12, axes=(3,)))
    dij = dij.set(csdl.slice[:,time_ind,:,:,2], value=csdl.norm(dpij[:,time_ind,:,:,2,:]+1.e-12, axes=(3,)))
    dij = dij.set(csdl.slice[:,time_ind,:,:,3], value=csdl.norm(dpij[:,time_ind,:,:,3,:]+1.e-12, axes=(3,)))
    surf_wake_mesh_dict['dij'] = dij

    mij = surf_wake_mesh_dict['mij']
    mij = mij.set(csdl.slice[:,time_ind,:,:,0], value=(dpij[:,time_ind,:,:,0,1])/(dpij[:,time_ind,:,:,0,0]+1.e-12))
    mij = mij.set(csdl.slice[:,time_ind,:,:,1], value=(dpij[:,time_ind,:,:,1,1])/(dpij[:,time_ind,:,:,1,0]+1.e-12))
    mij = mij.set(csdl.slice[:,time_ind,:,:,2], value=(dpij[:,time_ind,:,:,2,1])/(dpij[:,time_ind,:,:,2,0]+1.e-12))
    mij = mij.set(csdl.slice[:,time_ind,:,:,3], value=(dpij[:,time_ind,:,:,3,1])/(dpij[:,time_ind,:,:,3,0]+1.e-12))
    surf_wake_mesh_dict['mij'] = mij

    # nodal_vel = surf_wake_mesh_dict[key]['nodal_velocity']
    # surf_wake_mesh_dict[key]['coll_point_velocity'] = (nodal_vel[:,:,:-1,:-1,:]+nodal_vel[:,:,:-1,1:,:]+nodal_vel[:,:,1:,1:,:]+nodal_vel[:,:,1:,:-1,:]) / 4.

    return surf_wake_mesh_dict