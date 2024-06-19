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

    p1 = mesh[:,time_ind,:-1,:-1,:]
    p2 = mesh[:,time_ind,:-1,1:,:]
    p3 = mesh[:,time_ind,1:,1:,:]
    p4 = mesh[:,time_ind,1:,:-1,:]

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
    panel_x_vec = p4 - p1
    panel_y_vec = p2 - p1
    panel_normal_vec = csdl.cross(panel_x_vec, panel_y_vec, axis=3)
    panel_area = panel_area.set(csdl.slice[:,time_ind,:,:], value=csdl.norm(panel_normal_vec+1.e-12, axes=(3,)) / 2.)
    surf_wake_mesh_dict['panel_area'] = panel_area

    panel_x_dir = surf_wake_mesh_dict['panel_x_dir']
    panel_y_dir = surf_wake_mesh_dict['panel_y_dir']
    panel_normal = surf_wake_mesh_dict['panel_normal']

    panel_x_dir_val = panel_x_vec / csdl.expand(csdl.norm(panel_x_vec+1.e-12, axes=(3,)), panel_x_vec.shape, 'ijk->ijka')
    panel_y_dir_val = panel_y_vec / csdl.expand(csdl.norm(panel_y_vec+1.e-12, axes=(3,)), panel_y_vec.shape, 'ijk->ijka')
    panel_normal_val = panel_normal_vec / csdl.expand(csdl.norm(panel_normal_vec+1.e-12, axes=(3,)), panel_normal_vec.shape, 'ijk->ijka')

    panel_x_dir = panel_x_dir.set(csdl.slice[:,time_ind,:,:,:], value=panel_x_dir_val)
    panel_y_dir = panel_y_dir.set(csdl.slice[:,time_ind,:,:,:], value=panel_y_dir_val)
    panel_normal = panel_normal.set(csdl.slice[:,time_ind,:,:,:], value=panel_normal_val)

    surf_wake_mesh_dict['panel_x_dir'] = panel_x_dir
    surf_wake_mesh_dict['panel_y_dir'] = panel_y_dir
    surf_wake_mesh_dict['panel_normal'] = panel_normal

    return surf_wake_mesh_dict