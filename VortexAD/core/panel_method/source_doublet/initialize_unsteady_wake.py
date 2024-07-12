import numpy as np
import csdl_alpha as csdl 

from VortexAD.core.panel_method.source_doublet.wake_geometry import wake_geometry
from VortexAD.utils.atan2_switch import atan2_switch

def initialize_unsteady_wake(mesh_dict, num_nodes, dt, panel_fraction=0.25):
    surface_names = list(mesh_dict.keys())

    wake_mesh_dict = {}

    for i, surface_name in enumerate(surface_names):

        surf_wake_mesh_dict = {}

        surface_mesh = mesh_dict[surface_name]['mesh']
        mesh_velocity = mesh_dict[surface_name]['nodal_velocity']

        nt = surface_mesh.shape[1]
        ns = surface_mesh.shape[3]

        TE = (surface_mesh[:,:,0,:,:] + surface_mesh[:,:,-1,:,:])/2.
        TE_vel = (mesh_velocity[:,:,0,:,:] + mesh_velocity[:,:,-1,:,:])/2.
        init_wake_pos = TE + panel_fraction*TE_vel*dt

        nc_w = nt # we initialize small wake which is t = 0, so the wake MESH has +1 rows (+1 panels)

        # wake_mesh = csdl.Variable(shape=surface_mesh.shape[:2] + (nc_w,) + surface_mesh.shape[3:], value=0.)
        wake_mesh = csdl.Variable(shape=(num_nodes, nt, nc_w, ns, 3), value=0.)
        wake_mesh = wake_mesh.set(csdl.slice[:,:,0,:,:], value=TE)
        wake_mesh = wake_mesh.set(csdl.slice[:,:,1:,:,:], value=csdl.expand(TE, (num_nodes, nt, nc_w-1, ns, 3), 'ijkl->ijakl'))
        # wake_mesh = wake_mesh.set(csdl.slice[:,:,1:,:,:], value = csdl.expand(init_wake_pos, (num_nodes, nt, nc_w-1, ns, 3), 'ijkl->ijakl'))

        wake_mesh_velocity = csdl.Variable(shape=(num_nodes, nt, nc_w, ns, 3), value=0.)
        wake_mesh_velocity = wake_mesh_velocity.set(csdl.slice[:,:,0,:,:], value=TE_vel)
        # wake_mesh_velocity = wake_mesh_velocity.set(csdl.slice[:,:,1,:,:], value=TE_vel)

        surf_wake_mesh_dict['mesh'] = wake_mesh
        surf_wake_mesh_dict['nc'], surf_wake_mesh_dict['ns'] = nc_w, ns
        surf_wake_mesh_dict['num_panels'] = (nc_w-1)*(ns-1)
        surf_wake_mesh_dict['num_points'] = nc_w*ns

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
        
        wake_mesh_dict[surface_name] = surf_wake_mesh_dict


    return wake_mesh_dict

