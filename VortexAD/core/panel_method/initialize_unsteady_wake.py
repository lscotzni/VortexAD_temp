import numpy as np
import csdl_alpha as csdl 

from VortexAD.core.panel_method.wake_geometry import wake_geometry
from VortexAD.utils.atan2_switch import atan2_switch

def initialize_unsteady_wake(mesh_dict, num_nodes, dt, panel_fraction=0.25):
    surface_names = list(mesh_dict.keys())

    wake_mesh_dict = {}

    x_dir = np.array([1., 0., 0.,])
    y_dir = np.array([0., 1., 0.,])
    z_dir = np.array([0., 0., 1.,])

    for i, surface_name in enumerate(surface_names):

        surf_wake_mesh_dict = {}

        surface_mesh = mesh_dict[surface_name]['mesh']
        mesh_velocity = mesh_dict[surface_name]['nodal_velocity']

        nt = surface_mesh.shape[1]
        ns = surface_mesh.shape[3]

        TE = (surface_mesh[:,:,0,:,:] + surface_mesh[:,:,-1,:,:])/2.
        TE_vel = (mesh_velocity[:,:,0,:,:] + mesh_velocity[:,:,-1,:,:])/2.
        init_wake_pos = TE - panel_fraction*TE_vel*dt

        nc_w = nt+2 # we initialize small wake which is t = 0, so the wake MESH has +2 rows (+1 panels)

        # wake_mesh = csdl.Variable(shape=surface_mesh.shape[:2] + (nc_w,) + surface_mesh.shape[3:], value=0.)
        wake_mesh = csdl.Variable(shape=(num_nodes, nt, nc_w, ns, 3), value=0.)
        wake_mesh = wake_mesh.set(csdl.slice[:,:,0,:,:], value = TE)
        wake_mesh = wake_mesh.set(csdl.slice[:,:,1,:,:], value = init_wake_pos)

        wake_mesh_velocity = csdl.Variable(shape=(num_nodes, nt, nc_w, ns, 3), value=0.)
        wake_mesh_velocity = wake_mesh_velocity.set(csdl.slice[:,:,0,:,:], value=TE_vel)
        wake_mesh_velocity = wake_mesh_velocity.set(csdl.slice[:,:,1,:,:], value=TE_vel)

        surf_wake_mesh_dict['mesh'] = wake_mesh
        surf_wake_mesh_dict['nc'], surf_wake_mesh_dict['ns'] = nc_w, ns
        surf_wake_mesh_dict['num_panels'] = (nc_w-1)*(ns-1)

        # INITIALIZING OTHER WAKE MESH PARAMETERS
        surf_wake_mesh_dict['panel_center'] = csdl.Variable(shape=(num_nodes, nt, nc_w-1, ns-1, 3), value=0.)
        surf_wake_mesh_dict['panel_corners'] = csdl.Variable(shape=((num_nodes, nt, nc_w-1, ns-1, 4, 3)), value=0.)
        surf_wake_mesh_dict['panel_area'] = csdl.Variable(shape=(num_nodes, nt, nc_w-1, ns-1), value=0.)
        surf_wake_mesh_dict['panel_x_dir'] = csdl.Variable(shape=(num_nodes, nt, nc_w-1, ns-1, 3), value=0.)
        surf_wake_mesh_dict['panel_y_dir'] = csdl.Variable(shape=(num_nodes, nt, nc_w-1, ns-1, 3), value=0.)
        surf_wake_mesh_dict['panel_normal'] = csdl.Variable(shape=(num_nodes, nt, nc_w-1, ns-1, 3), value=0.)
        surf_wake_mesh_dict['dpij'] = csdl.Variable(value=np.zeros((num_nodes, nt, nc_w-1, ns-1, 4, 2)))
        surf_wake_mesh_dict['dij'] = csdl.Variable(value=np.zeros((num_nodes, nt, nc_w-1, ns-1, 4)))
        surf_wake_mesh_dict['mij'] = csdl.Variable(shape=(num_nodes, nt, nc_w-1, ns-1, 4), value=0.)

        surf_wake_mesh_dict['wake_nodal_velocity'] = wake_mesh_velocity

        surf_wake_mesh_dict = wake_geometry(surf_wake_mesh_dict, time_ind=0)
        

        # p1 = wake_mesh[:,0,:-1,:-1,:]
        # p2 = wake_mesh[:,0,:-1,1:,:]
        # p3 = wake_mesh[:,0,1:,1:,:]
        # p4 = wake_mesh[:,0,1:,:-1,:]

        # panel_center = csdl.Variable(shape=(num_nodes, nt, nt, ns-1, 3), value=0.)
        # panel_center = panel_center.set(csdl.slice[:,0,:,:,:], value=(p1+p2+p3+p4)/4.)
        # surf_wake_mesh_dict['panel_center'] = panel_center

        # panel_corners = csdl.Variable(shape=(panel_center.shape[:-1] + (4,3)), value=0.)
        # panel_corners = panel_corners.set(csdl.slice[:,0,:,:,0,:], value=p1)
        # panel_corners = panel_corners.set(csdl.slice[:,0,:,:,1,:], value=p2)
        # panel_corners = panel_corners.set(csdl.slice[:,0,:,:,2,:], value=p3)
        # panel_corners = panel_corners.set(csdl.slice[:,0,:,:,3,:], value=p4)
        # surf_wake_mesh_dict['panel_corners'] = panel_corners

        # panel_area = csdl.Variable(shape=panel_center.shape, value=0.)
        # panel_x_vec = p4 - p1
        # panel_y_vec = p2 - p1
        # panel_normal_vec = csdl.cross(panel_x_vec, panel_y_vec, axis=4)
        # panel_area_0 = csdl.norm(panel_normal_vec, axes=(4,)) / 2.
        # panel_area = panel_area.set(csdl.slice[:,0,:,:,:], value=panel_area_0)
        # surf_wake_mesh_dict['panel_area'] = panel_area

        # panel_x_dir = csdl.Variable(shape=panel_center.shape, value=0.)
        # panel_y_dir = csdl.Variable(shape=panel_center.shape, value=0.)
        # panel_normal = csdl.Variable(shape=panel_center.shape, value=0.)

        # panel_x_dir_0 = panel_x_vec / csdl.expand(csdl.norm(panel_x_vec, axes=(3,)), panel_x_vec.shape, 'ijk->ijka')
        # panel_y_dir_0 = panel_y_vec / csdl.expand(csdl.norm(panel_y_vec, axes=(3,)), panel_y_vec.shape, 'ijk->ijka')
        # panel_normal_0 = panel_normal_vec / csdl.expand(csdl.norm(panel_normal_vec, axes=(3,)), panel_normal_vec.shape, 'ijk->ijka')

        # panel_x_dir = panel_x_dir.set(csdl.slice[:,0,:,:,:], value=panel_x_dir_0)
        # panel_y_dir = panel_y_dir.set(csdl.slice[:,0,:,:,:], value=panel_y_dir_0)
        # panel_normal = panel_normal.set(csdl.slice[:,0,:,:,:], value=panel_normal_0)

        # surf_wake_mesh_dict['panel_x_dir'] = panel_x_dir
        # surf_wake_mesh_dict['panel_y_dir'] = panel_y_dir
        # surf_wake_mesh_dict['panel_normal'] = panel_normal

        # Px_normal = csdl.tensordot(panel_normal, x_dir, axes=([4],[0]))
        # Py_normal = csdl.tensordot(panel_normal, y_dir, axes=([4],[0]))
        # Pz_normal = csdl.tensordot(panel_normal, z_dir, axes=([4],[0]))

        # theta_x = atan2_switch(Py_normal, Pz_normal, scale=100.) # roll angle
        # theta_y = atan2_switch(-Px_normal, Pz_normal, scale=100.) # pitch angle

        # Px_x_dir = csdl.tensordot(panel_x_dir, x_dir, axes=([4],[0]))
        # Py_x_dir = csdl.tensordot(panel_x_dir, y_dir, axes=([4],[0]))

        # theta_z = atan2_switch(Py_x_dir, Px_x_dir, scale=100.)

        # dpij = csdl.Variable(value=np.zeros((panel_center.shape[:-1] + (4,2))))
        # dpij = dpij.set(csdl.slice[:,0,:,:,0,:], value=p2[:,0,:,:,:2]-p1[:,0,:,:,:2])
        # dpij = dpij.set(csdl.slice[:,0,:,:,1,:], value=p3[:,0,:,:,:2]-p2[:,0,:,:,:2])
        # dpij = dpij.set(csdl.slice[:,0,:,:,2,:], value=p4[:,0,:,:,:2]-p3[:,0,:,:,:2])
        # dpij = dpij.set(csdl.slice[:,0,:,:,3,:], value=p1[:,0,:,:,:2]-p4[:,0,:,:,:2])

        # surf_wake_mesh_dict['dpij'] = dpij

        # dij = csdl.Variable(value=np.zeros((panel_center.shape[:-1] + (4,))))
        # dij = dij.set(csdl.slice[:,0,:,:,0], value=csdl.norm(dpij[:,0,:,:,0,:]))
        # dij = dij.set(csdl.slice[:,0,:,:,1], value=csdl.norm(dpij[:,0,:,:,1,:]))
        # dij = dij.set(csdl.slice[:,0,:,:,2], value=csdl.norm(dpij[:,0,:,:,2,:]))
        # dij = dij.set(csdl.slice[:,0,:,:,3], value=csdl.norm(dpij[:,0,:,:,3,:]))

        # surf_wake_mesh_dict['dij'] = dij

        # mij = csdl.Variable(shape=dij.shape, value=0.)
        # mij = mij.set(csdl.slice[:,0,:,:,0], value=(dpij[:,0,:,:,0,1])/(dpij[:,0,:,:,0,0]+1.e-12))
        # mij = mij.set(csdl.slice[:,0,:,:,1], value=(dpij[:,0,:,:,1,1])/(dpij[:,0,:,:,1,0]+1.e-12))
        # mij = mij.set(csdl.slice[:,0,:,:,2], value=(dpij[:,0,:,:,2,1])/(dpij[:,0,:,:,2,0]+1.e-12))
        # mij = mij.set(csdl.slice[:,0,:,:,3], value=(dpij[:,0,:,:,3,1])/(dpij[:,0,:,:,3,0]+1.e-12))

        # surf_wake_mesh_dict['mij'] = mij
        
        wake_mesh_dict[surface_name] = surf_wake_mesh_dict


    return wake_mesh_dict

